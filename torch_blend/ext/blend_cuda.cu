#include "blend.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

template <typename scalar_t>
__global__ void blend_hwc_kernel(
    const scalar_t* __restrict__ img1,
    const scalar_t* __restrict__ img2,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t numel,
    int channels,
    float max_value) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }

    const int64_t mask_index = index / channels;
    const float alpha = static_cast<float>(mask[mask_index]) / max_value;
    const float value =
        static_cast<float>(img1[index]) * alpha
        + static_cast<float>(img2[index]) * (1.0f - alpha);
    output[index] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void blend_channel_first_kernel(
    const scalar_t* __restrict__ img1,
    const scalar_t* __restrict__ img2,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ output,
    int64_t spatial_size,
    int64_t sample_size,
    bool mask_is_batched,
    float max_value) {
    const int64_t spatial_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (spatial_index >= spatial_size) {
        return;
    }

    const int64_t batch_index = blockIdx.z;
    const int64_t channel_index = blockIdx.y;
    const int64_t image_index =
        batch_index * sample_size
        + channel_index * spatial_size
        + spatial_index;
    const int64_t mask_index =
        (mask_is_batched ? batch_index * spatial_size : 0)
        + spatial_index;

    const float alpha = static_cast<float>(mask[mask_index]) / max_value;
    const float value =
        static_cast<float>(img1[image_index]) * alpha
        + static_cast<float>(img2[image_index]) * (1.0f - alpha);
    output[image_index] = static_cast<scalar_t>(value);
}

}  // namespace

void blend_cuda(
    const torch::Tensor& img1,
    const torch::Tensor& img2,
    const torch::Tensor& mask,
    torch::Tensor& output,
    const BlendMetadata& metadata,
    float max_value,
    c10::optional<torch::Stream> stream) {
    cudaStream_t cuda_stream =
        c10::cuda::getCurrentCUDAStream(img1.device().index());
    if (stream.has_value()) {
        const auto requested_stream = stream.value();
        TORCH_CHECK(
            requested_stream.device().is_cuda(),
            "Provided stream must be a CUDA stream");
        TORCH_CHECK(
            requested_stream.device() == img1.device(),
            "Stream and tensors must be on the same device");
        cuda_stream = c10::cuda::CUDAStream(requested_stream).stream();
    }

    constexpr int threads = 256;

    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half,
        img1.scalar_type(),
        "blend_images_cuda",
        [&] {
            if (metadata.layout == BlendLayout::HWC) {
                const int blocks = static_cast<int>(
                    (metadata.numel + threads - 1) / threads);
                blend_hwc_kernel<scalar_t><<<
                    blocks,
                    threads,
                    0,
                    cuda_stream>>>(
                        img1.data_ptr<scalar_t>(),
                        img2.data_ptr<scalar_t>(),
                        mask.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        metadata.numel,
                        metadata.channels,
                        max_value);
            } else {
                const int64_t batch_size =
                    metadata.numel / metadata.sample_size;
                TORCH_CHECK(
                    batch_size <= 65535,
                    "CUDA batch size exceeds gridDim.z limit");
                TORCH_CHECK(
                    metadata.channels <= 65535,
                    "CUDA channel count exceeds gridDim.y limit");

                const dim3 grid(
                    static_cast<unsigned int>(
                        (metadata.spatial_size + threads - 1) / threads),
                    static_cast<unsigned int>(metadata.channels),
                    static_cast<unsigned int>(batch_size));
                blend_channel_first_kernel<scalar_t><<<
                    grid,
                    threads,
                    0,
                    cuda_stream>>>(
                        img1.data_ptr<scalar_t>(),
                        img2.data_ptr<scalar_t>(),
                        mask.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        metadata.spatial_size,
                        metadata.sample_size,
                        metadata.mask_is_batched,
                        max_value);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (!stream.has_value()) {
        C10_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    }
}
