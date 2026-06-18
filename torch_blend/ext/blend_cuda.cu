#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// KERNEL CUDA
template <typename scalar_t>
__global__ void blendImagesKernel(const scalar_t* img1, const scalar_t* img2, const scalar_t* mask, scalar_t* output, int width, int height, int channels, float max_val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_idx = (y * width + x) * channels;
        int mask_idx = y * width + x;

        float alpha = static_cast<float>(mask[mask_idx]) / max_val;
        float beta = 1.0f - alpha;

        for (int c = 0; c < channels; ++c) {
            int idx = pixel_idx + c;
            float val = static_cast<float>(img1[idx]) * alpha + static_cast<float>(img2[idx]) * beta;

            output[idx] = static_cast<scalar_t>(val);
        }
    }
}

// DISPATCHER FUNCTION
torch::Tensor blend_dispatcher(
    torch::Tensor img1, 
    torch::Tensor img2, 
    torch::Tensor mask, 
    c10::optional<torch::Stream> stream_opt) {
    
    // Input validation
    TORCH_CHECK(img1.dim() == 3 && img2.dim() == 3, "Images must be 3D (H, W, C)");
    TORCH_CHECK(mask.dim() == 2, "Mask must be 2D (H, W)");
    
    TORCH_CHECK(img1.dtype() == img2.dtype() && img1.dtype() == mask.dtype(), "All tensors must have the same dtype");
    
    int height = img1.size(0);
    int width = img1.size(1);
    int channels = img1.size(2);

    // Allocate output tensor
    torch::Tensor output = torch::empty_like(img1);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    float max_val = 255.0f;
    if (img1.is_floating_point()) {
        max_val = 1.0f;
    }

    if (img1.is_cuda()) {
        TORCH_CHECK(img2.is_cuda() && mask.is_cuda(), "All tensors must be on the same CUDA device");
        
        cudaStream_t cuda_stream = 0;
        if (stream_opt.has_value()) {
            auto stream = stream_opt.value();
            TORCH_CHECK(stream.device().is_cuda(), "Provided stream must be a CUDA stream.");
            TORCH_CHECK(stream.device() == img1.device(), "Stream and Tensors must be on the same device!");
            cuda_stream = c10::cuda::getCurrentCUDAStream(stream.device().index());
        }

        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, img1.scalar_type(), "blend_images_kernel", [&] {
            blendImagesKernel<scalar_t><<<gridDim, blockDim, 0, cuda_stream>>>(
                img1.data_ptr<scalar_t>(), 
                img2.data_ptr<scalar_t>(), 
                mask.data_ptr<scalar_t>(), 
                output.data_ptr<scalar_t>(), 
                width, height, channels, max_val
            );
        });

        if (!stream_opt.has_value()) {
            cudaStreamSynchronize(cuda_stream);
        }
        
    } 
    else if (img1.is_cpu()) {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, img1.scalar_type(), "blend_images_cpu", [&] {
            auto img1_ptr = img1.data_ptr<scalar_t>();
            auto img2_ptr = img2.data_ptr<scalar_t>();
            auto mask_ptr = mask.data_ptr<scalar_t>();
            auto out_ptr = output.data_ptr<scalar_t>();

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int pixel_idx = (y * width + x) * channels;
                    int mask_idx = y * width + x;
                    float alpha = static_cast<float>(mask_ptr[mask_idx]) / max_val;
                    float beta = 1.0f - alpha;
                    for (int c = 0; c < channels; ++c) {
                        int idx = pixel_idx + c;
                        float val = static_cast<float>(img1_ptr[idx]) * alpha + static_cast<float>(img2_ptr[idx]) * beta;
                        out_ptr[idx] = static_cast<scalar_t>(val);
                    }
                }
            }
        });
    }
    else {
        TORCH_CHECK(false, "Unsupported device type");
    }

    return output;
}

// PYBIND11 MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blend", &blend_dispatcher, "Blend 2 images with a mask (Multi-dtype, Auto Sync/Async)",
          py::arg("img1"), 
          py::arg("img2"), 
          py::arg("mask"), 
          py::arg("stream") = py::none());
}