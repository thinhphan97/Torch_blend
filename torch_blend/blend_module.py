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
            img1 (torch.Tensor): Background image. Shape: (H, W, C). 
                Dtype: uint8, float32, or float16. (If float, values should be in [0.0, 1.0]).
            img2 (torch.Tensor): Foreground image. Must match img1's shape and dtype.
            mask (torch.Tensor): Grayscale mask. Shape: (H, W). Must match img1's dtype.
            stream (Optional[torch.cuda.Stream], optional): 
                If None: Execution is SYNCHRONOUS (blocks CPU until GPU finishes).
                If stream provided: Execution is ASYNCHRONOUS (returns immediately).
                Ignored for CPU tensors. Defaults to None.
            
        Returns:
            torch.Tensor: The blended image. Shape: (H, W, C). Dtype matches the input tensors.
            
        Raises:
            ValueError: If tensors are not on the same device, have mismatched shapes, or mismatched dtypes.
            TypeError: If stream is not a torch.cuda.Stream instance.
            
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
                
        # Call the underlying C++ extension
        return torch_blend_cuda.blend(img1, img2, mask, stream)
