#include <torch/extension.h>

#include "blend.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "blend",
        &blend,
        "Blend two tensors using an alpha mask",
        py::arg("img1"),
        py::arg("img2"),
        py::arg("mask"),
        py::arg("layout"),
        py::arg("mask_is_batched"),
        py::arg("height"),
        py::arg("width"),
        py::arg("blend_mode"),
        py::arg("stream") = py::none());
}
