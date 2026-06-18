#include "blend.h"

namespace {

BlendMetadata make_metadata(
    const torch::Tensor& image,
    int layout,
    bool mask_is_batched,
    int64_t height,
    int64_t width) {
    TORCH_CHECK(layout >= 0 && layout <= 2, "Invalid layout code");
    TORCH_CHECK(height >= 0 && width >= 0, "Image dimensions must be non-negative");

    const auto blend_layout = static_cast<BlendLayout>(layout);
    const int channels =
        blend_layout == BlendLayout::HWC
        ? image.size(2)
        : image.size(image.dim() - 3);
    const int64_t spatial_size = height * width;

    return {
        image.numel(),
        spatial_size,
        static_cast<int64_t>(channels) * spatial_size,
        channels,
        blend_layout,
        mask_is_batched,
    };
}

}  // namespace

torch::Tensor blend(
    torch::Tensor img1,
    torch::Tensor img2,
    torch::Tensor mask,
    int layout,
    bool mask_is_batched,
    int64_t height,
    int64_t width,
    c10::optional<torch::Stream> stream) {
    TORCH_CHECK(img1.dim() == 3 || img1.dim() == 4, "Images must be 3D or 4D");
    TORCH_CHECK(img1.sizes() == img2.sizes(), "Image shapes must match");
    TORCH_CHECK(
        img1.dtype() == img2.dtype() && img1.dtype() == mask.dtype(),
        "All tensors must have the same dtype");
    TORCH_CHECK(
        img1.device() == img2.device() && img1.device() == mask.device(),
        "All tensors must be on the same device");
    TORCH_CHECK(img1.is_contiguous(), "img1 must be contiguous");
    TORCH_CHECK(img2.is_contiguous(), "img2 must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");

    auto output = torch::empty_like(img1);
    if (output.numel() == 0) {
        return output;
    }

    const auto metadata = make_metadata(
        img1,
        layout,
        mask_is_batched,
        height,
        width);
    const float max_value = img1.is_floating_point() ? 1.0f : 255.0f;

    if (img1.is_cuda()) {
        blend_cuda(img1, img2, mask, output, metadata, max_value, stream);
    } else if (img1.is_cpu()) {
        blend_cpu(img1, img2, mask, output, metadata, max_value);
    } else {
        TORCH_CHECK(false, "Unsupported device type");
    }

    return output;
}
