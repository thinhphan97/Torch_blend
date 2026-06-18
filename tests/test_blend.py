import pytest
import torch
import warnings
from torch_blend import ImageBlender


def blend_reference_float(img1, img2, mask, device='cpu', layout='HWC'):
    """Compute the expected blend in float32 on the requested device."""
    img1_d = img1.to(device).float()
    img2_d = img2.to(device).float()
    mask_d = mask.to(device).float()
    
    max_val = 255.0 if img1.dtype == torch.uint8 else 1.0
    alpha = mask_d / max_val
    if layout == 'HWC':
        alpha = alpha.unsqueeze(-1)
    elif layout == 'CHW':
        if alpha.dim() == 3:
            alpha = alpha.squeeze(0)
        alpha = alpha.unsqueeze(0)
    elif layout in {'BCHW', 'NCHW'}:
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(0).unsqueeze(0)
        elif alpha.dim() == 3:
            alpha = alpha.unsqueeze(1)
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

    def test_deterministic_small_input(self):
        """Verify exact output for a small input with fixed values."""
        img1 = torch.tensor([[[0], [100]], [[200], [255]]], dtype=torch.uint8)
        img2 = torch.tensor([[[255], [200]], [[100], [0]]], dtype=torch.uint8)
        mask = torch.tensor([[0, 255], [128, 64]], dtype=torch.uint8)

        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(img1, img2, mask).to(torch.uint8)

        assert torch.equal(result, expected)

    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_supported_channel_counts(self, channels):
        """Verify blending for grayscale, RGB, and RGBA channel counts."""
        img1 = torch.full((4, 5, channels), 40, dtype=torch.uint8)
        img2 = torch.full((4, 5, channels), 200, dtype=torch.uint8)
        mask = torch.full((4, 5), 128, dtype=torch.uint8)

        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(img1, img2, mask).to(torch.uint8)

        assert torch.equal(result, expected)

    def test_output_properties(self, sample_tensors):
        """Verify that output preserves shape, dtype, device, and contiguity."""
        img1, img2, mask = sample_tensors

        result = ImageBlender.blend(img1, img2, mask)

        assert result.shape == img1.shape
        assert result.dtype == img1.dtype
        assert result.device == img1.device
        assert result.is_contiguous()

    def test_non_contiguous_inputs(self, sample_tensors):
        """Verify that non-contiguous images and masks are blended correctly."""
        img1, img2, mask = sample_tensors
        img1 = img1.transpose(0, 1)
        img2 = img2.transpose(0, 1)
        mask = mask.transpose(0, 1)
        assert not img1.is_contiguous()
        assert not img2.is_contiguous()
        assert not mask.is_contiguous()

        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(img1, img2, mask).to(torch.uint8)

        assert torch.equal(result, expected)

    @pytest.mark.parametrize("shape", [(0, 4, 3), (4, 0, 3)])
    def test_empty_images(self, shape):
        """Verify that empty spatial dimensions return an empty output."""
        img1 = torch.empty(shape, dtype=torch.uint8)
        img2 = torch.empty(shape, dtype=torch.uint8)
        mask = torch.empty(shape[:2], dtype=torch.uint8)

        result = ImageBlender.blend(img1, img2, mask)

        assert result.shape == img1.shape
        assert result.numel() == 0


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_stream_argument_outside_context(self):
        """Verify that the supplied CUDA stream is used outside its context manager."""
        img1 = torch.zeros((8, 8, 3), dtype=torch.float32, device="cuda")
        img2 = torch.zeros_like(img1)
        mask = torch.ones((8, 8), dtype=torch.float32, device="cuda")
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            img1.fill_(1.0)

        result = ImageBlender.blend(img1, img2, mask, stream=stream)
        stream.synchronize()

        assert torch.equal(result, torch.ones_like(result))

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


class TestTensorLayouts:
    """Verify PyTorch-native CHW and BCHW tensor layouts."""

    def test_chw_auto_detection(self):
        """Infer CHW from a three-dimensional image and spatial mask."""
        img1 = torch.randint(0, 256, (3, 4, 5), dtype=torch.uint8)
        img2 = torch.randint(0, 256, (3, 4, 5), dtype=torch.uint8)
        mask = torch.randint(0, 256, (4, 5), dtype=torch.uint8)

        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(img1, img2, mask, layout='CHW').to(torch.uint8)

        assert torch.equal(result, expected)

    def test_chw_singleton_channel_mask(self):
        """Accept a CHW mask with shape (1, H, W)."""
        img1 = torch.rand((3, 4, 5), dtype=torch.float32)
        img2 = torch.rand((3, 4, 5), dtype=torch.float32)
        mask = torch.rand((1, 4, 5), dtype=torch.float32)

        result = ImageBlender.blend(img1, img2, mask, layout='CHW')
        expected = blend_reference_float(img1, img2, mask, layout='CHW')

        torch.testing.assert_close(result, expected)

    def test_explicit_chw_resolves_ambiguous_shape(self):
        """Use the layout argument when a 3D shape is valid as HWC and CHW."""
        img1 = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
        img2 = torch.flip(img1, dims=(0,))
        mask = torch.tensor(
            [[0.0, 0.25, 0.5], [0.75, 1.0, 0.25], [0.5, 0.75, 1.0]],
            dtype=torch.float32,
        )

        result = ImageBlender.blend(img1, img2, mask, layout='CHW')
        expected = blend_reference_float(img1, img2, mask, layout='CHW')

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("mask_layout", ["BHW", "B1HW", "HW"])
    def test_bchw_batch_blend(self, mask_layout):
        """Blend BCHW batches with per-sample or shared masks."""
        img1 = torch.rand((2, 3, 4, 5), dtype=torch.float32)
        img2 = torch.rand((2, 3, 4, 5), dtype=torch.float32)
        if mask_layout == "BHW":
            mask = torch.rand((2, 4, 5), dtype=torch.float32)
        elif mask_layout == "B1HW":
            mask = torch.rand((2, 1, 4, 5), dtype=torch.float32)
        else:
            mask = torch.rand((4, 5), dtype=torch.float32)

        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(img1, img2, mask, layout='BCHW')

        torch.testing.assert_close(result, expected)

    def test_non_contiguous_bchw_inputs(self):
        """Blend non-contiguous BCHW tensors after internal normalization."""
        img1 = torch.rand((2, 3, 5, 4), dtype=torch.float32).transpose(2, 3)
        img2 = torch.rand((2, 3, 5, 4), dtype=torch.float32).transpose(2, 3)
        mask = torch.rand((2, 5, 4), dtype=torch.float32).transpose(1, 2)
        assert not img1.is_contiguous()
        assert not img2.is_contiguous()
        assert not mask.is_contiguous()

        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(img1, img2, mask, layout='BCHW')

        torch.testing.assert_close(result, expected)
        assert result.is_contiguous()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bchw_cuda_blend(self):
        """Verify batched BCHW blending on CUDA."""
        img1 = torch.rand((2, 3, 8, 8), dtype=torch.float32, device='cuda')
        img2 = torch.rand_like(img1)
        mask = torch.rand((2, 8, 8), dtype=torch.float32, device='cuda')

        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(img1, img2, mask, device='cuda', layout='BCHW')

        torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)


class TestVectorizedCuda:
    """Verify CUDA vectorized fast paths and scalar fallback behavior."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    def test_hwc_four_channel_vectorized_blend(self, dtype):
        """Verify uchar4 and float4 HWC blending for four-channel images."""
        if dtype == torch.uint8:
            img1 = torch.randint(0, 256, (16, 20, 4), dtype=dtype, device='cuda')
            img2 = torch.randint(0, 256, (16, 20, 4), dtype=dtype, device='cuda')
            mask = torch.randint(0, 256, (16, 20), dtype=dtype, device='cuda')
        else:
            img1 = torch.rand((16, 20, 4), dtype=dtype, device='cuda')
            img2 = torch.rand((16, 20, 4), dtype=dtype, device='cuda')
            mask = torch.rand((16, 20), dtype=dtype, device='cuda')

        result = ImageBlender.blend(img1, img2, mask, layout='HWC')
        expected = blend_reference_float(
            img1,
            img2,
            mask,
            device='cuda',
            layout='HWC',
        )

        if dtype == torch.uint8:
            torch.testing.assert_close(
                result.float(),
                expected.to(dtype).float(),
                rtol=0,
                atol=1,
            )
        else:
            torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("mask_layout", ["BHW", "HW"])
    def test_bchw_vectorized_blend(self, dtype, mask_layout):
        """Verify vectorized BCHW blending with batched and shared masks."""
        shape = (2, 3, 8, 12)
        if dtype == torch.uint8:
            img1 = torch.randint(0, 256, shape, dtype=dtype, device='cuda')
            img2 = torch.randint(0, 256, shape, dtype=dtype, device='cuda')
            mask_shape = (2, 8, 12) if mask_layout == "BHW" else (8, 12)
            mask = torch.randint(0, 256, mask_shape, dtype=dtype, device='cuda')
        else:
            img1 = torch.rand(shape, dtype=dtype, device='cuda')
            img2 = torch.rand(shape, dtype=dtype, device='cuda')
            mask_shape = (2, 8, 12) if mask_layout == "BHW" else (8, 12)
            mask = torch.rand(mask_shape, dtype=dtype, device='cuda')

        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(
            img1,
            img2,
            mask,
            device='cuda',
            layout='BCHW',
        )

        if dtype == torch.uint8:
            torch.testing.assert_close(
                result.float(),
                expected.to(dtype).float(),
                rtol=0,
                atol=1,
            )
        else:
            torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    def test_bchw_scalar_fallback_for_unaligned_plane_size(self, dtype):
        """Verify scalar fallback when H*W is not divisible by four."""
        shape = (2, 3, 3, 5)
        if dtype == torch.uint8:
            img1 = torch.randint(0, 256, shape, dtype=dtype, device='cuda')
            img2 = torch.randint(0, 256, shape, dtype=dtype, device='cuda')
            mask = torch.randint(0, 256, (2, 3, 5), dtype=dtype, device='cuda')
        else:
            img1 = torch.rand(shape, dtype=dtype, device='cuda')
            img2 = torch.rand(shape, dtype=dtype, device='cuda')
            mask = torch.rand((2, 3, 5), dtype=dtype, device='cuda')

        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(
            img1,
            img2,
            mask,
            device='cuda',
            layout='BCHW',
        )

        if dtype == torch.uint8:
            torch.testing.assert_close(
                result.float(),
                expected.to(dtype).float(),
                rtol=0,
                atol=1,
            )
        else:
            torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)


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
        
        with pytest.raises(ValueError, match="Cannot infer"):
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

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float64])
    def test_unsupported_dtype(self, dtype):
        """Reject dtypes outside uint8, float16, and float32."""
        img1 = torch.zeros((4, 4, 3), dtype=dtype)
        img2 = torch.zeros_like(img1)
        mask = torch.zeros((4, 4), dtype=dtype)

        with pytest.raises(TypeError, match="Unsupported dtype"):
            ImageBlender.blend(img1, img2, mask)

    @pytest.mark.parametrize("shape", [(4, 4), (1, 2, 3, 4, 5)])
    def test_invalid_image_dimensions(self, shape):
        """Reject images that are neither 3D nor 4D tensors."""
        img1 = torch.zeros(shape, dtype=torch.uint8)
        img2 = torch.zeros_like(img1)
        mask = torch.zeros((4, 4), dtype=torch.uint8)

        with pytest.raises(ValueError, match="Images must be 3D.*or 4D"):
            ImageBlender.blend(img1, img2, mask)

    def test_mask_spatial_shape_mismatch(self):
        """Reject a two-dimensional mask with incorrect spatial dimensions."""
        img1 = torch.zeros((4, 5, 3), dtype=torch.uint8)
        img2 = torch.zeros_like(img1)
        mask = torch.zeros((4, 4), dtype=torch.uint8)

        with pytest.raises(ValueError, match="HWC inputs require a mask"):
            ImageBlender.blend(img1, img2, mask, layout='HWC')

    def test_invalid_layout(self, sample_tensors):
        """Reject unknown explicit layout names."""
        img1, img2, mask = sample_tensors

        with pytest.raises(ValueError, match="Layout must be one of"):
            ImageBlender.blend(img1, img2, mask, layout='NHWC')

    def test_invalid_bchw_mask_shape(self):
        """Reject batched masks that do not match batch or spatial dimensions."""
        img1 = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
        img2 = torch.zeros_like(img1)
        mask = torch.zeros((3, 4, 5), dtype=torch.float32)

        with pytest.raises(ValueError, match="BCHW masks must have shape"):
            ImageBlender.blend(img1, img2, mask)


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

    def test_float16_cpu_blend(self, sample_tensors):
        """Verify normalized float16 CPU blending."""
        img1, img2, mask = sample_tensors
        img1 = (img1.float() / 255.0).half()
        img2 = (img2.float() / 255.0).half()
        mask = (mask.float() / 255.0).half()

        result = ImageBlender.blend(img1, img2, mask)
        expected = blend_reference_float(img1, img2, mask)

        torch.testing.assert_close(result.float(), expected, rtol=0, atol=1e-3)

    def test_float_mask_extrapolation(self):
        """Verify that float masks outside [0, 1] perform linear extrapolation."""
        img1 = torch.ones((1, 2, 1), dtype=torch.float32)
        img2 = torch.zeros_like(img1)
        mask = torch.tensor([[-0.5, 1.5]], dtype=torch.float32)

        result = ImageBlender.blend(img1, img2, mask)

        torch.testing.assert_close(
            result,
            torch.tensor([[[-0.5], [1.5]]], dtype=torch.float32),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_output_properties(self, sample_tensors):
        """Verify that CUDA output preserves shape, dtype, device, and contiguity."""
        img1, img2, mask = (tensor.cuda() for tensor in sample_tensors)

        result = ImageBlender.blend(img1, img2, mask)

        assert result.shape == img1.shape
        assert result.dtype == img1.dtype
        assert result.device == img1.device
        assert result.is_contiguous()

    def test_dtype_mismatch_tensors(self, sample_tensors):
        """Reject inputs when the second image has a mismatched dtype."""
        img1, img2, mask = sample_tensors
        img2_float = img2.float()
        with pytest.raises(ValueError, match="All tensors must have the same dtype"):
            ImageBlender.blend(img1, img2_float, mask)
