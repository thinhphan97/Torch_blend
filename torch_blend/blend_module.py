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
        stream: Optional[torch.cuda.Stream] = None
    ) -> torch.Tensor:
        """
        Blends two images using a mask. Supports CPU and CUDA, with multi-dtype capabilities.
        
        Args:
            img1 (torch.Tensor): Background image with shape (H, W, C).
                Supported dtypes are uint8, float16, and float32.
            img2 (torch.Tensor): Foreground image. Must match img1's shape and dtype.
            mask (torch.Tensor): Grayscale mask. Shape: (H, W). Must match img1's dtype.
                uint8 masks use 255 as full opacity. Floating-point masks use 1.0;
                values outside [0.0, 1.0] perform linear extrapolation.
            stream (Optional[torch.cuda.Stream], optional): 
                If None: Execution is SYNCHRONOUS (blocks CPU until GPU finishes).
                If stream provided: Execution is ASYNCHRONOUS (returns immediately).
                The supplied stream is used directly and does not need to be the
                current stream. Ignored for CPU tensors. Defaults to None.
            
        Returns:
            torch.Tensor: A contiguous blended image with shape (H, W, C).
                Dtype and device match the input tensors. Empty spatial inputs
                return an empty tensor with matching metadata.
            
        Raises:
            ValueError: If tensors are not on the same device, have mismatched shapes, or mismatched dtypes.
            TypeError: If the dtype is unsupported or stream is not a torch.cuda.Stream instance.

        Notes:
            Non-contiguous inputs are converted to contiguous tensors before the
            native CPU or CUDA implementation is called.
            
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

        # Validate devices
        if img1.device != img2.device or img1.device != mask.device:
            raise ValueError("All tensors must be located on the same device.")
            
        # Validate
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes must match. Got {img1.shape} and {img2.shape}")
        if img1.dim() != 3 or mask.dim() != 2:
            raise ValueError("Images must be 3D and Mask must be 2D")
        if img1.shape[:2] != mask.shape:
            raise ValueError(f"Mask spatial shape must match images. Got {mask.shape} for images of shape {img1.shape[:2]}")
        if img1.dtype != img2.dtype or img1.dtype != mask.dtype:
            raise ValueError(f"All tensors must have the same dtype. Got {img1.dtype}, {img2.dtype}, {mask.dtype}")
        if img1.dtype not in supported_dtypes:
            raise TypeError(
                f"Unsupported dtype {img1.dtype}. "
                "Supported dtypes are torch.uint8, torch.float16, and torch.float32."
            )
            
        is_cpu = img1.is_cpu
        
        # Handle stream validation and warnings
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
                
        # The native implementation operates on contiguous HWC/HW memory.
        return torch_blend_cuda.blend(
            img1.contiguous(),
            img2.contiguous(),
            mask.contiguous(),
            stream,
        )
