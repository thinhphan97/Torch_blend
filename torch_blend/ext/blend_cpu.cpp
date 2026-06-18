#include "blend.h"

void blend_cpu(
    const torch::Tensor& img1,
    const torch::Tensor& img2,
    const torch::Tensor& mask,
    torch::Tensor& output,
    const BlendMetadata& metadata,
    float max_value) {
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half,
        img1.scalar_type(),
        "blend_images_cpu",
        [&] {
            const auto* img1_data = img1.data_ptr<scalar_t>();
            const auto* img2_data = img2.data_ptr<scalar_t>();
            const auto* mask_data = mask.data_ptr<scalar_t>();
            auto* output_data = output.data_ptr<scalar_t>();

            for (int64_t index = 0; index < metadata.numel; ++index) {
                const int64_t mask_index = blend_mask_index(index, metadata);
                const float alpha =
                    static_cast<float>(mask_data[mask_index]) / max_value;
                const float value =
                    static_cast<float>(img1_data[index]) * alpha
                    + static_cast<float>(img2_data[index]) * (1.0f - alpha);
                output_data[index] = static_cast<scalar_t>(value);
            }
        });
}
