#pragma once

#include <c10/macros/Macros.h>

#include <cstdint>

enum class BlendLayout : int {
    HWC = 0,
    CHW = 1,
    BCHW = 2,
};

struct BlendMetadata {
    int64_t numel;
    int64_t spatial_size;
    int64_t sample_size;
    int channels;
    BlendLayout layout;
    bool mask_is_batched;
};

C10_HOST_DEVICE inline int64_t blend_mask_index(
    int64_t element_index,
    const BlendMetadata& metadata) {
    int64_t spatial_index;
    int64_t batch_index = 0;

    if (metadata.layout == BlendLayout::HWC) {
        spatial_index = element_index / metadata.channels;
    } else {
        spatial_index = element_index % metadata.spatial_size;
        if (metadata.layout == BlendLayout::BCHW) {
            batch_index = element_index / metadata.sample_size;
        }
    }

    if (metadata.layout == BlendLayout::BCHW && metadata.mask_is_batched) {
        return batch_index * metadata.spatial_size + spatial_index;
    }
    return spatial_index;
}
