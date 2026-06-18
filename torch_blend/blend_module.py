import torch
import warnings
from typing import Optional
import torch_blend._torch_blend_cuda as torch_blend_cuda


class ImageBlender:
    """
    Image Blender utility using PyTorch and CUDA.
    Behavior is automatically synchronous if no stream is provided,
    and asynchronous if a CUDA stream is provided.
    """
    
    @staticmethod
    def blend(
        img1: torch.Tensor, 
        img2: torch.Tensor, 
        mask: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
        layout: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Blends two images using a mask. Supports CPU and CUDA, with multi-dtype capabilities.
        
        Args:
            img1 (torch.Tensor): Background image with shape (H, W, C),
                (C, H, W), or (B, C, H, W).
                Supported dtypes are uint8, float16, and float32.
            img2 (torch.Tensor): Foreground image. Must match img1's shape and dtype.
            mask (torch.Tensor): Mask matching the spatial dimensions. Batched
                inputs accept (B, H, W), (B, 1, H, W), or a shared (H, W) mask.
                uint8 masks use 255 as full opacity. Floating-point masks use 1.0;
                values outside [0.0, 1.0] perform linear extrapolation.
            stream (Optional[torch.cuda.Stream], optional): 
                If None: Execution is SYNCHRONOUS (blocks CPU until GPU finishes).
                If stream provided: Execution is ASYNCHRONOUS (returns immediately).
                The supplied stream is used directly and does not need to be the
                current stream. Ignored for CPU tensors. Defaults to None.
            layout (Optional[str], optional): Explicit image layout: "HWC",
                "CHW", "BCHW", or "NCHW". By default, the layout is inferred
                from image and mask shapes. Ambiguous 3D inputs prefer HWC for
                backward compatibility.
            
        Returns:
            torch.Tensor: A contiguous blended image with the same shape as img1.
                Dtype and device match the input tensors. Empty spatial inputs
                return an empty tensor with matching metadata.
            
        Raises:
            ValueError: If devices, shapes, dtypes, layouts, or mask dimensions are invalid.
            TypeError: If the dtype is unsupported or stream is not a torch.cuda.Stream instance.

        Notes:
            Non-contiguous inputs are converted to contiguous tensors. For 3D
            inputs whose shapes are valid as both HWC and CHW, pass layout
            explicitly to select CHW.
            
        Example (Sync - uint8):
            >>> img1 = torch.randint(0, 255, (1080, 1920, 3), dtype=torch.uint8)
            >>> result = ImageBlender.blend(img1, img2, mask) # Blocks until done
            
        Example (Async - float32):
            >>> img1_f = img1.float() / 255.0
            >>> my_stream = torch.cuda.Stream()
            >>> result = ImageBlender.blend(img1_f, img2_f, mask_f, stream=my_stream)
            >>> # Do other CPU work...
            >>> my_stream.synchronize() # Wait when needed
        """
        supported_dtypes = {torch.uint8, torch.float16, torch.float32}
        layout_codes = {"HWC": 0, "CHW": 1, "BCHW": 2, "NCHW": 2}

        if img1.device != img2.device or img1.device != mask.device:
            raise ValueError("All tensors must be located on the same device.")
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes must match. Got {img1.shape} and {img2.shape}")
        if img1.dtype != img2.dtype or img1.dtype != mask.dtype:
            raise ValueError(f"All tensors must have the same dtype. Got {img1.dtype}, {img2.dtype}, {mask.dtype}")
        if img1.dtype not in supported_dtypes:
            raise TypeError(
                f"Unsupported dtype {img1.dtype}. "
                "Supported dtypes are torch.uint8, torch.float16, and torch.float32."
            )

        normalized_layout = layout.upper() if layout is not None else None
        if normalized_layout is not None and normalized_layout not in layout_codes:
            raise ValueError("Layout must be one of: HWC, CHW, BCHW, or NCHW.")

        if img1.dim() == 3:
            hwc_mask_shape = tuple(img1.shape[:2])
            chw_mask_shape = tuple(img1.shape[1:])
            mask_shape = tuple(mask.shape)
            is_hwc = mask.dim() == 2 and mask_shape == hwc_mask_shape
            is_chw = (
                (mask.dim() == 2 and mask_shape == chw_mask_shape)
                or (mask.dim() == 3 and mask.shape[0] == 1 and tuple(mask.shape[1:]) == chw_mask_shape)
            )

            if normalized_layout is None:
                if is_hwc:
                    normalized_layout = "HWC"
                elif is_chw:
                    normalized_layout = "CHW"
                else:
                    raise ValueError(
                        f"Cannot infer a valid HWC or CHW layout from image shape "
                        f"{tuple(img1.shape)} and mask shape {tuple(mask.shape)}."
                    )
            elif normalized_layout == "HWC" and not is_hwc:
                raise ValueError(f"HWC inputs require a mask with shape {hwc_mask_shape}.")
            elif normalized_layout == "CHW" and not is_chw:
                raise ValueError(f"CHW inputs require a mask with shape {chw_mask_shape} or (1, H, W).")
            elif normalized_layout in {"BCHW", "NCHW"}:
                raise ValueError("BCHW/NCHW layout requires 4D image tensors.")

            height, width = (
                (img1.shape[0], img1.shape[1])
                if normalized_layout == "HWC"
                else (img1.shape[1], img1.shape[2])
            )
            mask_is_batched = False
        elif img1.dim() == 4:
            if normalized_layout not in {None, "BCHW", "NCHW"}:
                raise ValueError("4D image tensors require BCHW or NCHW layout.")
            normalized_layout = "BCHW"
            batch, _, height, width = img1.shape
            spatial_shape = (height, width)
            is_shared_mask = mask.dim() == 2 and tuple(mask.shape) == spatial_shape
            is_batched_mask = (
                (mask.dim() == 3 and tuple(mask.shape) == (batch, height, width))
                or (
                    mask.dim() == 4
                    and tuple(mask.shape) == (batch, 1, height, width)
                )
            )
            if not is_shared_mask and not is_batched_mask:
                raise ValueError(
                    "BCHW masks must have shape (H, W), (B, H, W), or (B, 1, H, W)."
                )
            mask_is_batched = is_batched_mask
        else:
            raise ValueError("Images must be 3D (HWC/CHW) or 4D (BCHW/NCHW).")

        is_cpu = img1.is_cpu
        if stream is not None:
            if is_cpu:
                warnings.warn(
                    "A stream was provided, but the tensors are on CPU. "
                    "The stream argument will be ignored.", 
                    RuntimeWarning
                )
                stream = None 
            elif not isinstance(stream, torch.cuda.Stream):
                raise TypeError("Stream must be an instance of torch.cuda.Stream or None.")

        return torch_blend_cuda.blend(
            img1.contiguous(),
            img2.contiguous(),
            mask.contiguous(),
            layout_codes[normalized_layout],
            mask_is_batched,
            height,
            width,
            stream,
        )
