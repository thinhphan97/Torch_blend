import pytest
import torch
import warnings
from torch_blend import ImageBlender


def blend_reference_float(img1, img2, mask, device='cpu'):
    """Compute the expected blend in float32 on the requested device."""
    img1_d = img1.to(device).float()
    img2_d = img2.to(device).float()
    mask_d = mask.to(device).float()
    
    max_val = 255.0 if img1.dtype == torch.uint8 else 1.0
    alpha = (mask_d / max_val).unsqueeze(-1)
    beta = 1.0 - alpha
    
    return img1_d * alpha + img2_d * beta


class TestCoreFunctionality:
    """Verify core blending behavior on CPU and CUDA devices."""

    def test_cpu_blend_correctness(self, sample_tensors):
        """Verify that uint8 CPU output matches the PyTorch reference."""
        img1, img2, mask = sample_tensors
        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(img1, img2, mask).to(torch.uint8)
        assert torch.equal(result, expected), "CPU blend output mismatch"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_sync_blend_correctness(self, sample_tensors):
        """Verify synchronous uint8 CUDA blending within quantization tolerance."""
        img1, img2, mask = sample_tensors
        img1_gpu, img2_gpu, mask_gpu = img1.cuda(), img2.cuda(), mask.cuda()
        
        result = ImageBlender.blend(img1_gpu, img2_gpu, mask_gpu)
        expected = blend_reference_float(img1, img2, mask, device='cuda').to(img1_gpu.dtype)
        assert result.dtype == img1_gpu.dtype
        torch.testing.assert_close(
            result.float(),
            expected.float(),
            rtol=0,
            atol=1,
            msg="GPU sync blend output mismatch",
        )

    def test_edge_cases_alpha(self, edge_case_tensors):
        """Verify that transparent and opaque masks select the correct image."""
        img1, img2, mask = edge_case_tensors
        result = ImageBlender.blend(img1, img2, mask)
        assert torch.all(result[:, :result.shape[1]//2] == 200), "Failed at mask=0"
        assert torch.all(result[:, result.shape[1]//2:] == 100), "Failed at mask=255"


class TestStreamAndAsync:
    """Verify CUDA stream handling and CPU stream fallback behavior."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_async_with_stream(self, sample_tensors):
        """Verify asynchronous CUDA blending on a user-provided stream."""
        img1, img2, mask = sample_tensors
        img1_gpu, img2_gpu, mask_gpu = img1.cuda(), img2.cuda(), mask.cuda()
        
        my_stream = torch.cuda.Stream()
        with torch.cuda.stream(my_stream):
            result = ImageBlender.blend(img1_gpu, img2_gpu, mask_gpu, stream=my_stream)
        
        my_stream.synchronize()
        
        expected = blend_reference_float(img1, img2, mask, device='cuda').to(img1_gpu.dtype)
        assert result.dtype == img1_gpu.dtype
        torch.testing.assert_close(
            result.float(),
            expected.float(),
            rtol=0,
            atol=1,
            msg="GPU async blend output mismatch",
        )

    def test_cpu_stream_warning(self, sample_tensors):
        """Verify that CPU blending ignores a stream and emits a warning."""
        img1, img2, mask = sample_tensors
        dummy_stream = torch.cuda.Stream() if torch.cuda.is_available() else object()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ImageBlender.blend(img1, img2, mask, stream=dummy_stream)
            assert len(w) > 0
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "ignored" in str(w[-1].message)
            
        expected = blend_reference_float(img1, img2, mask).to(torch.uint8)
        assert torch.equal(result, expected), "CPU blend with dummy stream failed"


class TestInputValidation:
    """Verify that invalid tensor and stream inputs are rejected."""

    def test_dtype_mismatch(self, sample_tensors):
        """Reject inputs when the first image has a mismatched dtype."""
        img1, img2, mask = sample_tensors
        img1_float = img1.float()
        with pytest.raises(ValueError, match="All tensors must have the same dtype"):
            ImageBlender.blend(img1_float, img2, mask)

    def test_shape_mismatch_images(self, sample_tensors):
        """Reject images with different shapes."""
        img1, img2, mask = sample_tensors
        img2_wrong_shape = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8)
        
        with pytest.raises(ValueError, match="shapes must match"):
            ImageBlender.blend(img1, img2_wrong_shape, mask)

    def test_wrong_mask_dimensions(self, sample_tensors):
        """Reject masks that are not two-dimensional."""
        img1, img2, mask = sample_tensors
        mask_3d = torch.randint(0, 256, (64, 64, 1), dtype=torch.uint8)
        
        with pytest.raises(ValueError, match="Mask must be 2D"):
            ImageBlender.blend(img1, img2, mask_3d)

    def test_device_mismatch(self, sample_tensors):
        """Reject tensors located on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        img1, img2, mask = sample_tensors
        img1_gpu = img1.cuda()
        
        with pytest.raises(ValueError, match="All tensors must be located on the same device"):
            ImageBlender.blend(img1_gpu, img2, mask)

    def test_invalid_stream_type(self, sample_tensors):
        """Reject stream arguments that are not torch.cuda.Stream instances."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        img1, img2, mask = sample_tensors
        img1_gpu, img2_gpu, mask_gpu = img1.cuda(), img2.cuda(), mask.cuda()
        
        with pytest.raises(TypeError, match="Stream must be an instance of torch.cuda.Stream"):
            ImageBlender.blend(img1_gpu, img2_gpu, mask_gpu, stream="invalid_stream_string")


class TestMultiDtype:
    """Verify blending behavior across supported tensor dtypes."""

    def test_float32_blend_correctness(self, sample_tensors):
        """Verify normalized float32 CPU blending against the reference."""
        img1, img2, mask = sample_tensors
        img1_f = (img1.float() / 255.0)
        img2_f = (img2.float() / 255.0)
        mask_f = (mask.float() / 255.0)
        
        result = ImageBlender.blend(img1_f, img2_f, mask_f)
        expected = blend_reference_float(img1_f, img2_f, mask_f)
        assert torch.allclose(result, expected, atol=1e-6), "Float32 blend output mismatch"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_float16_gpu_blend(self, sample_tensors):
        """Verify float16 CUDA blending within half-precision tolerance."""
        img1, img2, mask = sample_tensors
        img1_h = (img1.float() / 255.0).half().cuda()
        img2_h = (img2.float() / 255.0).half().cuda()
        mask_h = (mask.float() / 255.0).half().cuda()
        
        result = ImageBlender.blend(img1_h, img2_h, mask_h)
        expected = blend_reference_float(img1_h, img2_h, mask_h, device='cuda')
        assert torch.allclose(result.float(), expected, atol=1e-3), "Float16 GPU blend output mismatch"

    def test_dtype_mismatch_tensors(self, sample_tensors):
        """Reject inputs when the second image has a mismatched dtype."""
        img1, img2, mask = sample_tensors
        img2_float = img2.float()
        with pytest.raises(ValueError, match="All tensors must have the same dtype"):
            ImageBlender.blend(img1, img2_float, mask)
