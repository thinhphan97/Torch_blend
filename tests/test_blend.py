import pytest
import torch
import warnings
from torch_blend import ImageBlender


def cpu_blend_reference(img1, img2, mask):
    alpha = (mask.float() / 255.0).unsqueeze(-1)  # (H, W, 1)
    beta = 1.0 - alpha
    return torch.round(img1.float() * alpha + img2.float() * beta).to(torch.uint8)


class TestCoreFunctionality:
    """Test the mathematical correctness of the blend"""

    def test_cpu_blend_correctness(self, sample_tensors):
        """Verify CPU blend matches pure PyTorch logic"""
        img1, img2, mask = sample_tensors
        result = ImageBlender.blend(img1, img2, mask)
        expected = cpu_blend_reference(img1, img2, mask)
        assert torch.equal(result, expected), "CPU blend output mismatch"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_sync_blend_correctness(self, sample_tensors):
        """Verify GPU sync blend matches pure PyTorch logic"""
        img1, img2, mask = sample_tensors
        img1_gpu, img2_gpu, mask_gpu = img1.cuda(), img2.cuda(), mask.cuda()
        
        result = ImageBlender.blend(img1_gpu, img2_gpu, mask_gpu)
        expected = cpu_blend_reference(img1, img2, mask).cuda()
        assert torch.equal(result, expected), "GPU sync blend output mismatch"

    def test_edge_cases_alpha(self, edge_case_tensors):
        """Test boundaries: mask=0 (img2) and mask=255 (img1)"""
        img1, img2, mask = edge_case_tensors
        result = ImageBlender.blend(img1, img2, mask)
        
        # Where mask == 0, output should be img2
        assert torch.all(result[:, :result.shape[1]//2] == 200), "Failed at mask=0"
        # Where mask == 255, output should be img1
        assert torch.all(result[:, result.shape[1]//2:] == 100), "Failed at mask=255"


class TestStreamAndAsync:
    """Test Async/Sync behaviors based on streams"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_async_with_stream(self, sample_tensors):
        """Verify async execution does not block and stream sync works"""
        img1, img2, mask = sample_tensors
        img1_gpu, img2_gpu, mask_gpu = img1.cuda(), img2.cuda(), mask.cuda()
        
        my_stream = torch.cuda.Stream()
        
        with torch.cuda.stream(my_stream):
            # Should return immediately
            result = ImageBlender.blend(img1_gpu, img2_gpu, mask_gpu, stream=my_stream)
        
        # Wait for async task to finish
        my_stream.synchronize()
        
        expected = cpu_blend_reference(img1, img2, mask).cuda()
        assert torch.equal(result, expected), "GPU async blend output mismatch"

    def test_cpu_stream_warning(self, sample_tensors):
        """Verify that passing a stream to CPU tensors triggers a warning and ignores it"""
        img1, img2, mask = sample_tensors
        
        # Create a dummy stream object just to pass it in
        dummy_stream = torch.cuda.Stream() if torch.cuda.is_available() else object()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Should not crash, should warn, and should return correct CPU result
            result = ImageBlender.blend(img1, img2, mask, stream=dummy_stream)
            
            # Check if warning was raised
            assert len(w) > 0
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "ignored" in str(w[-1].message)
            
        expected = cpu_blend_reference(img1, img2, mask)
        assert torch.equal(result, expected), "CPU blend with dummy stream failed"


class TestInputValidation:
    """Test error handling for invalid inputs"""

    def test_dtype_mismatch(self, sample_tensors):
        """Should fail if tensors are not uint8"""
        img1, img2, mask = sample_tensors
        img1_float = img1.float() # Wrong dtype
        
        with pytest.raises(RuntimeError, match="Tensors must be uint8"):
            ImageBlender.blend(img1_float, img2, mask)

    def test_shape_mismatch_images(self, sample_tensors):
        """Should fail if img1 and img2 have different spatial dimensions"""
        img1, img2, mask = sample_tensors
        img2_wrong_shape = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8)
        
        with pytest.raises(ValueError, match="shapes must match"):
            ImageBlender.blend(img1, img2_wrong_shape, mask)

    def test_wrong_mask_dimensions(self, sample_tensors):
        """Should fail if mask is not 2D"""
        img1, img2, mask = sample_tensors
        mask_3d = torch.randint(0, 256, (64, 64, 1), dtype=torch.uint8)
        
        with pytest.raises(ValueError, match="Mask must be 2D"):
            ImageBlender.blend(img1, img2, mask_3d)

    def test_device_mismatch(self, sample_tensors):
        """Should fail if tensors are on different devices"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        img1, img2, mask = sample_tensors
        img1_gpu = img1.cuda()
        
        with pytest.raises(ValueError, match="All tensors must be located on the same device"):
            ImageBlender.blend(img1_gpu, img2, mask)

    def test_invalid_stream_type(self, sample_tensors):
        """Should fail if stream is not a torch.cuda.Stream instance"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        img1, img2, mask = sample_tensors
        img1_gpu, img2_gpu, mask_gpu = img1.cuda(), img2.cuda(), mask.cuda()
        
        with pytest.raises(TypeError, match="Stream must be an instance of torch.cuda.Stream"):
            ImageBlender.blend(img1_gpu, img2_gpu, mask_gpu, stream="invalid_stream_string")
