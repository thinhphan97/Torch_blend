#include "blend.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace {

bool is_aligned(const void* pointer, std::uintptr_t alignment) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

template <BlendMode mode>
__device__ unsigned char blend_uint8_value(
    unsigned char img1,
    unsigned char img2,
    float alpha) {
    return static_cast<unsigned char>(
        compose_blend_value<mode>(
            static_cast<float>(img1),
            static_cast<float>(img2),
            alpha,
            255.0f));
}

template <BlendMode mode>
__device__ float blend_float_value(float img1, float img2, float alpha) {
    return compose_blend_value<mode>(img1, img2, alpha, 1.0f);
}

template <typename scalar_t, BlendMode mode>
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
    const float value = compose_blend_value<mode>(
        static_cast<float>(img1[index]),
        static_cast<float>(img2[index]),
        alpha,
        max_value);
    output[index] = static_cast<scalar_t>(value);
}

template <BlendMode mode>
__global__ void blend_hwc_uchar4_kernel(
    const uchar4* __restrict__ img1,
    const uchar4* __restrict__ img2,
    const unsigned char* __restrict__ mask,
    uchar4* __restrict__ output,
    int64_t pixel_count) {
    const int64_t pixel_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel_index >= pixel_count) {
        return;
    }

    const float alpha = static_cast<float>(mask[pixel_index]) / 255.0f;
    const uchar4 img1_value = img1[pixel_index];
    const uchar4 img2_value = img2[pixel_index];

    output[pixel_index] = make_uchar4(
        blend_uint8_value<mode>(img1_value.x, img2_value.x, alpha),
        blend_uint8_value<mode>(img1_value.y, img2_value.y, alpha),
        blend_uint8_value<mode>(img1_value.z, img2_value.z, alpha),
        blend_uint8_value<mode>(img1_value.w, img2_value.w, alpha));
}

template <BlendMode mode>
__global__ void blend_hwc_float4_kernel(
    const float4* __restrict__ img1,
    const float4* __restrict__ img2,
    const float* __restrict__ mask,
    float4* __restrict__ output,
    int64_t pixel_count) {
    const int64_t pixel_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel_index >= pixel_count) {
        return;
    }

    const float alpha = mask[pixel_index];
    const float4 img1_value = img1[pixel_index];
    const float4 img2_value = img2[pixel_index];

    output[pixel_index] = make_float4(
        blend_float_value<mode>(img1_value.x, img2_value.x, alpha),
        blend_float_value<mode>(img1_value.y, img2_value.y, alpha),
        blend_float_value<mode>(img1_value.z, img2_value.z, alpha),
        blend_float_value<mode>(img1_value.w, img2_value.w, alpha));
}

template <typename scalar_t, BlendMode mode>
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
    const float value = compose_blend_value<mode>(
        static_cast<float>(img1[image_index]),
        static_cast<float>(img2[image_index]),
        alpha,
        max_value);
    output[image_index] = static_cast<scalar_t>(value);
}

template <BlendMode mode>
__global__ void blend_channel_first_uchar4_kernel(
    const uchar4* __restrict__ img1,
    const uchar4* __restrict__ img2,
    const uchar4* __restrict__ mask,
    uchar4* __restrict__ output,
    int64_t vectors_per_plane,
    int64_t vectors_per_sample,
    bool mask_is_batched) {
    const int64_t vector_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector_index >= vectors_per_plane) {
        return;
    }

    const int64_t batch_index = blockIdx.z;
    const int64_t channel_index = blockIdx.y;
    const int64_t image_index =
        batch_index * vectors_per_sample
        + channel_index * vectors_per_plane
        + vector_index;
    const int64_t mask_index =
        (mask_is_batched ? batch_index * vectors_per_plane : 0)
        + vector_index;

    const uchar4 img1_value = img1[image_index];
    const uchar4 img2_value = img2[image_index];
    const uchar4 mask_value = mask[mask_index];

    output[image_index] = make_uchar4(
        blend_uint8_value<mode>(
            img1_value.x,
            img2_value.x,
            static_cast<float>(mask_value.x) / 255.0f),
        blend_uint8_value<mode>(
            img1_value.y,
            img2_value.y,
            static_cast<float>(mask_value.y) / 255.0f),
        blend_uint8_value<mode>(
            img1_value.z,
            img2_value.z,
            static_cast<float>(mask_value.z) / 255.0f),
        blend_uint8_value<mode>(
            img1_value.w,
            img2_value.w,
            static_cast<float>(mask_value.w) / 255.0f));
}

template <BlendMode mode>
__global__ void blend_channel_first_float4_kernel(
    const float4* __restrict__ img1,
    const float4* __restrict__ img2,
    const float4* __restrict__ mask,
    float4* __restrict__ output,
    int64_t vectors_per_plane,
    int64_t vectors_per_sample,
    bool mask_is_batched) {
    const int64_t vector_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector_index >= vectors_per_plane) {
        return;
    }

    const int64_t batch_index = blockIdx.z;
    const int64_t channel_index = blockIdx.y;
    const int64_t image_index =
        batch_index * vectors_per_sample
        + channel_index * vectors_per_plane
        + vector_index;
    const int64_t mask_index =
        (mask_is_batched ? batch_index * vectors_per_plane : 0)
        + vector_index;

    const float4 img1_value = img1[image_index];
    const float4 img2_value = img2[image_index];
    const float4 mask_value = mask[mask_index];

    output[image_index] = make_float4(
        blend_float_value<mode>(img1_value.x, img2_value.x, mask_value.x),
        blend_float_value<mode>(img1_value.y, img2_value.y, mask_value.y),
        blend_float_value<mode>(img1_value.z, img2_value.z, mask_value.z),
        blend_float_value<mode>(img1_value.w, img2_value.w, mask_value.w));
}

template <BlendMode mode>
void launch_blend_cuda(
    const torch::Tensor& img1,
    const torch::Tensor& img2,
    const torch::Tensor& mask,
    torch::Tensor& output,
    const BlendMetadata& metadata,
    float max_value,
    cudaStream_t cuda_stream) {
    constexpr int threads = 256;
    bool launched = false;
    int64_t batch_size = 1;

    if (metadata.layout != BlendLayout::HWC) {
        batch_size = metadata.numel / metadata.sample_size;
        TORCH_CHECK(
            batch_size <= 65535,
            "CUDA batch size exceeds gridDim.z limit");
        TORCH_CHECK(
            metadata.channels <= 65535,
            "CUDA channel count exceeds gridDim.y limit");
    }

    if (metadata.layout == BlendLayout::HWC && metadata.channels == 4) {
        const int64_t pixel_count = metadata.numel / 4;
        const int blocks =
            static_cast<int>((pixel_count + threads - 1) / threads);

        if (
            img1.scalar_type() == at::ScalarType::Byte
            && is_aligned(img1.data_ptr<unsigned char>(), alignof(uchar4))
            && is_aligned(img2.data_ptr<unsigned char>(), alignof(uchar4))
            && is_aligned(output.data_ptr<unsigned char>(), alignof(uchar4))) {
            blend_hwc_uchar4_kernel<mode><<<blocks, threads, 0, cuda_stream>>>(
                reinterpret_cast<const uchar4*>(img1.data_ptr<unsigned char>()),
                reinterpret_cast<const uchar4*>(img2.data_ptr<unsigned char>()),
                mask.data_ptr<unsigned char>(),
                reinterpret_cast<uchar4*>(output.data_ptr<unsigned char>()),
                pixel_count);
            launched = true;
        } else if (
            img1.scalar_type() == at::ScalarType::Float
            && is_aligned(img1.data_ptr<float>(), alignof(float4))
            && is_aligned(img2.data_ptr<float>(), alignof(float4))
            && is_aligned(output.data_ptr<float>(), alignof(float4))) {
            blend_hwc_float4_kernel<mode><<<blocks, threads, 0, cuda_stream>>>(
                reinterpret_cast<const float4*>(img1.data_ptr<float>()),
                reinterpret_cast<const float4*>(img2.data_ptr<float>()),
                mask.data_ptr<float>(),
                reinterpret_cast<float4*>(output.data_ptr<float>()),
                pixel_count);
            launched = true;
        }
    }

    if (
        !launched
        && metadata.layout != BlendLayout::HWC
        && metadata.spatial_size % 4 == 0) {
        const int64_t vectors_per_plane = metadata.spatial_size / 4;
        const int64_t vectors_per_sample = metadata.sample_size / 4;
        const dim3 grid(
            static_cast<unsigned int>(
                (vectors_per_plane + threads - 1) / threads),
            static_cast<unsigned int>(metadata.channels),
            static_cast<unsigned int>(batch_size));

        if (
            img1.scalar_type() == at::ScalarType::Byte
            && is_aligned(img1.data_ptr<unsigned char>(), alignof(uchar4))
            && is_aligned(img2.data_ptr<unsigned char>(), alignof(uchar4))
            && is_aligned(mask.data_ptr<unsigned char>(), alignof(uchar4))
            && is_aligned(output.data_ptr<unsigned char>(), alignof(uchar4))) {
            blend_channel_first_uchar4_kernel<mode><<<
                grid,
                threads,
                0,
                cuda_stream>>>(
                    reinterpret_cast<const uchar4*>(
                        img1.data_ptr<unsigned char>()),
                    reinterpret_cast<const uchar4*>(
                        img2.data_ptr<unsigned char>()),
                    reinterpret_cast<const uchar4*>(
                        mask.data_ptr<unsigned char>()),
                    reinterpret_cast<uchar4*>(
                        output.data_ptr<unsigned char>()),
                    vectors_per_plane,
                    vectors_per_sample,
                    metadata.mask_is_batched);
            launched = true;
        } else if (
            img1.scalar_type() == at::ScalarType::Float
            && is_aligned(img1.data_ptr<float>(), alignof(float4))
            && is_aligned(img2.data_ptr<float>(), alignof(float4))
            && is_aligned(mask.data_ptr<float>(), alignof(float4))
            && is_aligned(output.data_ptr<float>(), alignof(float4))) {
            blend_channel_first_float4_kernel<mode><<<
                grid,
                threads,
                0,
                cuda_stream>>>(
                    reinterpret_cast<const float4*>(img1.data_ptr<float>()),
                    reinterpret_cast<const float4*>(img2.data_ptr<float>()),
                    reinterpret_cast<const float4*>(mask.data_ptr<float>()),
                    reinterpret_cast<float4*>(output.data_ptr<float>()),
                    vectors_per_plane,
                    vectors_per_sample,
                    metadata.mask_is_batched);
            launched = true;
        }
    }

    if (!launched) {
        AT_DISPATCH_ALL_TYPES_AND(
            at::ScalarType::Half,
            img1.scalar_type(),
            "blend_images_cuda_scalar",
            [&] {
                if (metadata.layout == BlendLayout::HWC) {
                    const int blocks = static_cast<int>(
                        (metadata.numel + threads - 1) / threads);
                    blend_hwc_kernel<scalar_t, mode><<<
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
                    const dim3 grid(
                        static_cast<unsigned int>(
                            (metadata.spatial_size + threads - 1) / threads),
                        static_cast<unsigned int>(metadata.channels),
                        static_cast<unsigned int>(batch_size));
                    blend_channel_first_kernel<scalar_t, mode><<<
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
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void blend_cuda(
    const torch::Tensor& img1,
    const torch::Tensor& img2,
    const torch::Tensor& mask,
    torch::Tensor& output,
    const BlendMetadata& metadata,
    float max_value,
    BlendMode blend_mode,
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

    switch (blend_mode) {
        case BlendMode::Linear:
            launch_blend_cuda<BlendMode::Linear>(
                img1, img2, mask, output, metadata, max_value, cuda_stream);
            break;
        case BlendMode::Multiply:
            launch_blend_cuda<BlendMode::Multiply>(
                img1, img2, mask, output, metadata, max_value, cuda_stream);
            break;
        case BlendMode::Screen:
            launch_blend_cuda<BlendMode::Screen>(
                img1, img2, mask, output, metadata, max_value, cuda_stream);
            break;
        case BlendMode::Overlay:
            launch_blend_cuda<BlendMode::Overlay>(
                img1, img2, mask, output, metadata, max_value, cuda_stream);
            break;
    }

    if (!stream.has_value()) {
        C10_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    }
}
