#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// CUDA KERNEL: Blends two images using a mask
__global__ void blendImagesKernel(const unsigned char* img1, const unsigned char* img2, const unsigned char* mask, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_idx = (y * width + x) * channels;
        int mask_idx = y * width + x;

        float alpha = mask[mask_idx] / 255.0f;
        float beta = 1.0f - alpha;

        for (int c = 0; c < channels; ++c) {
            int idx = pixel_idx + c;
            float val = img1[idx] * alpha + img2[idx] * beta;
            output[idx] = static_cast<unsigned char>(val + 0.5f);
        }
    }
}

// DISPATCHER FUNCTION
// If stream is provided -> Async execution.
// If stream is None -> Synchronous execution.
torch::Tensor blend_dispatcher(
    torch::Tensor img1, 
    torch::Tensor img2, 
    torch::Tensor mask, 
    c10::optional<torch::Stream> stream_opt) {
    
    // Input validation
    TORCH_CHECK(img1.dim() == 3 && img2.dim() == 3, "Images must be 3D (H, W, C)");
    TORCH_CHECK(mask.dim() == 2, "Mask must be 2D (H, W)");
    TORCH_CHECK(img1.dtype() == torch::kUInt8 && img2.dtype() == torch::kUInt8 && mask.dtype() == torch::kUInt8, "Tensors must be uint8");
    
    int height = img1.size(0);
    int width = img1.size(1);
    int channels = img1.size(2);

    torch::Tensor output = torch::empty_like(img1);

    auto img1_ptr = img1.data_ptr<unsigned char>();
    auto img2_ptr = img2.data_ptr<unsigned char>();
    auto mask_ptr = mask.data_ptr<unsigned char>();
    auto out_ptr = output.data_ptr<unsigned char>();

    if (img1.is_cuda()) {
        TORCH_CHECK(img2.is_cuda() && mask.is_cuda(), "All tensors must be on the same CUDA device");
        
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        if (stream_opt.has_value()) {
            // ASYNC MODE: User provided a stream
            auto stream = stream_opt.value();
            TORCH_CHECK(stream.device().is_cuda(), "Provided stream must be a CUDA stream.");
            TORCH_CHECK(stream.device() == img1.device(), "Stream and Tensors must be on the same device!");
            
            // Lấy cudaStream_t trực tiếp từ PyTorch Stream mà không cần CUDAStreamGuard
            cudaStream_t cuda_stream = c10::cuda::getCurrentCUDAStream(stream.device().index());
            
            // Launch kernel on the user stream (Returns immediately)
            blendImagesKernel<<<gridDim, blockDim, 0, cuda_stream>>>(img1_ptr, img2_ptr, mask_ptr, out_ptr, width, height, channels);
        } else {
            // SYNC MODE: No stream provided. Run on default stream and wait.
            blendImagesKernel<<<gridDim, blockDim>>>(img1_ptr, img2_ptr, mask_ptr, out_ptr, width, height, channels);
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
        }
        
    } 
    else if (img1.is_cpu()) {
        // CPU is inherently synchronous, stream is ignored if passed.
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int pixel_idx = (y * width + x) * channels;
                int mask_idx = y * width + x;
                float alpha = mask_ptr[mask_idx] / 255.0f;
                float beta = 1.0f - alpha;
                for (int c = 0; c < channels; ++c) {
                    int idx = pixel_idx + c;
                    float val = img1_ptr[idx] * alpha + img2_ptr[idx] * beta;
                    out_ptr[idx] = static_cast<unsigned char>(val + 0.5f);
                }
            }
        }
    }
    else {
        TORCH_CHECK(false, "Unsupported device type");
    }

    return output;
}

// PYBIND11 MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blend", &blend_dispatcher, "Blend 2 images with a mask (Auto Sync/Async based on stream)",
          py::arg("img1"), 
          py::arg("img2"), 
          py::arg("mask"), 
          py::arg("stream") = py::none());
}