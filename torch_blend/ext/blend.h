#pragma once

#include <torch/extension.h>

#include "blend_common.h"
#include "blend_modes.h"

torch::Tensor blend(
    torch::Tensor img1,
    torch::Tensor img2,
    torch::Tensor mask,
    int layout,
    bool mask_is_batched,
    int64_t height,
    int64_t width,
    int blend_mode,
    c10::optional<torch::Stream> stream);

void blend_cpu(
    const torch::Tensor& img1,
    const torch::Tensor& img2,
    const torch::Tensor& mask,
    torch::Tensor& output,
    const BlendMetadata& metadata,
    float max_value,
    BlendMode blend_mode);

void blend_cuda(
    const torch::Tensor& img1,
    const torch::Tensor& img2,
    const torch::Tensor& mask,
    torch::Tensor& output,
    const BlendMetadata& metadata,
    float max_value,
    BlendMode blend_mode,
    c10::optional<torch::Stream> stream);
