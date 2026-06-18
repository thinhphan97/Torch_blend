#pragma once

#include <c10/macros/Macros.h>

enum class BlendMode : int {
    Linear = 0,
    Multiply = 1,
    Screen = 2,
    Overlay = 3,
};

template <BlendMode mode>
C10_HOST_DEVICE inline float apply_blend_mode(
    float img1,
    float img2,
    float max_value) {
    if constexpr (mode == BlendMode::Linear) {
        return img1;
    } else if constexpr (mode == BlendMode::Multiply) {
        return img1 * img2 / max_value;
    } else if constexpr (mode == BlendMode::Screen) {
        return max_value - ((max_value - img1) * (max_value - img2) / max_value);
    } else {
        const float midpoint = max_value * 0.5f;
        if (img2 <= midpoint) {
            return 2.0f * img1 * img2 / max_value;
        }
        return max_value - (2.0f * (max_value - img1) * (max_value - img2) / max_value);
    }
}

template <BlendMode mode>
C10_HOST_DEVICE inline float compose_blend_value(
    float img1,
    float img2,
    float alpha,
    float max_value) {
    const float mode_value = apply_blend_mode<mode>(img1, img2, max_value);
    return mode_value * alpha + img2 * (1.0f - alpha);
}
