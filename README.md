# Torch Blend Package

A high-performance PyTorch C++/CUDA extension for blending two images using a mask. 
This package provides a custom PyTorch operator that runs natively on CUDA (with a CPU fallback), supporting both **synchronous** and **asynchronous** execution via custom CUDA streams.

## ✨ Features

- **Multi-Dtype Support**: Supports `uint8`, `float32`, and `float16` tensors on CPU and CUDA. Other dtypes are rejected explicitly.
- **Contiguous Native Operations**: Directly accesses PyTorch tensor memory for contiguous inputs. Non-contiguous inputs are converted automatically before dispatch.
- **Auto Sync/Async**: Runs synchronously if no stream is provided and asynchronously on the exact `torch.cuda.Stream` passed by the caller.
- **Device-Aware**: Falls back to a pure C++ CPU loop if tensors are on CPU, with friendly warnings if streams are misused.
- **Flexible Tensor Layouts**: Supports OpenCV-style `HWC`, PyTorch-style `CHW`, and batched `BCHW`/`NCHW` tensors.
- **Batch Processing**: Applies shared or per-sample masks to an entire image or feature-map batch.
- **Layout-Specific CUDA Kernels**: Uses a coalesced linear kernel for `HWC` and maps `BCHW` batches to `gridDim.z`, channels to `gridDim.y`, and spatial pixels to `threadIdx.x`.
- **Vectorized CUDA Access**: Uses `uchar4` and `float4` fast paths when tensor layout, alignment, and spatial dimensions permit four-value memory transactions.
- **Advanced Blend Modes**: Supports linear, multiply, screen, and overlay operations with mask-controlled compositing.
- **Edge-Case Handling**: Preserves output metadata and safely handles empty spatial dimensions.
- **Pythonic API**: Comes with a clean, typed Python wrapper with full docstrings and IDE autocomplete support.

## 📦 Project Structure

```text
torch_blend_package/
├── setup.py                 # Build configuration using torch.utils.cpp_extension
├── README.md
├── torch_blend/             # Source code
│   ├── __init__.py
│   ├── blend_module.py      # Python wrapper (Type hints, validation)
│   └── ext/
│       ├── bindings.cpp     # Pybind11 module definition
│       ├── blend.h          # Native extension public declarations
│       ├── blend_common.h   # Shared layout metadata and index helpers
│       ├── blend_modes.h    # Shared CPU/CUDA blend mode formulas
│       ├── blend.cpp        # Device dispatcher and common validation
│       ├── blend_cpu.cpp    # CPU backend
│       └── blend_cuda.cu    # CUDA kernel and CUDA backend
└── tests/                   # Unit tests
    ├── __init__.py
    ├── conftest.py          # Pytest fixtures
    └── test_blend.py        # Test cases (including Multi-Dtype tests)
```

## 🛠️ Installation & Build

### 1. Prerequisites (Conda Environment Setup)

> **⚠️ CRITICAL NOTE:** To build PyTorch C++/CUDA extensions successfully, the CUDA version used by **PyTorch** must match the CUDA version of the **NVCC compiler** (`cuda-toolkit`).

Create a Conda environment and install matching PyTorch and CUDA toolkit versions. This example uses **CUDA 12.8**:

```bash
# Create and activate conda environment
conda create -n torch_blend_env python=3.10 -y
conda activate torch_blend_env

# Install PyTorch with a specific CUDA version (e.g., cu128)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install CUDA toolkit (nvcc) matching the SAME version (e.g., 12.8)
conda install nvidia::cuda-toolkit==12.8.2 -y

# Install build tools
pip install setuptools wheel pytest ninja
```

Verify that `nvcc` and PyTorch CUDA versions match:
```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

If the versions differ, the extension build fails with a CUDA version mismatch.

### 2. Building the Package

Clone the repository and build the extension. You have two options to build:

**Option 1: Build in-place (Development Mode)**
Compiles the `.cu` file and drops the `.so` library right next to your source code. Best for active development.

```bash
git clone https://github.com/thinhphan97/Torch_blend.git
cd Torch_blend

# Build the extension in-place
python setup.py build_ext --inplace
```

**Option 2: Build a Wheel Package (Distribution Mode)**
Compiles the code and packages it into a distributable `.whl` file located in the `dist/` folder.

```bash
git clone https://github.com/thinhphan97/Torch_blend.git
cd Torch_blend

# Build the wheel package
python setup.py bdist_wheel

# Install the generated wheel into your environment
pip install dist/torch_blend-*.whl
```

## 🧪 Running Tests

The test suite covers deterministic and randomized blending, CPU/CUDA execution,
custom streams, supported dtypes and channel counts, non-contiguous tensors,
empty inputs, `HWC`/`CHW`/`BCHW` layouts, batch masks, advanced blend modes,
output metadata, and input validation.

```bash
conda activate torch_blend_env
pytest tests/ -v
```

## 🚀 Usage & Examples

### 1. Standard Image Blending (uint8 - OpenCV Style)
By default, the function blocks the CPU until the GPU finishes processing. This is ideal for image processing scripts.

```python
import torch
from torch_blend import ImageBlender
import cv2

# Load images using OpenCV (H, W, C) and convert to PyTorch Tensors
img1_cv = cv2.imread("image1.jpg")
img2_cv = cv2.imread("image2.jpg")
mask_cv = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img1 = torch.from_numpy(img1_cv).to(device)
img2 = torch.from_numpy(img2_cv).to(device)
mask = torch.from_numpy(mask_cv).to(device)

# Synchronous Blend
result = ImageBlender.blend(img1, img2, mask)

cv2.imwrite("result_sync.jpg", result.cpu().numpy())
```

### 2. Deep Learning Pipeline (CHW / BCHW)
Floating-point images and masks are interpreted directly; they are not normalized
by the package. For standard alpha blending, use values in `[0.0, 1.0]`.
Mask values outside this range are allowed and perform linear extrapolation.

```python
import torch
from torch_blend import ImageBlender

# PyTorch-standard batched images: (B, C, H, W)
img1 = torch.rand(8, 3, 512, 512, device="cuda")
img2 = torch.rand_like(img1)
mask = torch.rand(8, 512, 512, device="cuda")

# Asynchronous Blend on GPU using float16 (half precision) for speed
my_stream = torch.cuda.Stream()

# The stream is used directly, even outside a stream context manager.
result_async = ImageBlender.blend(
    img1.half(),
    img2.half(),
    mask.half(),
    stream=my_stream,
    mode="screen",
)

# Explicitly wait for the GPU to finish
my_stream.synchronize()
```

### 3. CPU Fallback & Warnings
If you pass a stream but the tensors are on the CPU, the package will not crash. It will issue a `RuntimeWarning` and safely ignore the stream.

```python
import warnings
import torch
from torch_blend import ImageBlender

img1 = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
img2 = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
mask = torch.randint(0, 255, (64, 64), dtype=torch.uint8)

dummy_stream = torch.cuda.Stream()

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = ImageBlender.blend(img1, img2, mask, stream=dummy_stream)
    
    print(f"Warning raised: {w[0].message}")
```

## 📖 API Reference

### `ImageBlender.blend(img1, img2, mask, stream=None, layout=None, mode="linear")`

| Parameter | Type | Description |
|-----------|------|-------------|
| `img1` | `torch.Tensor` | Source or blend layer. Shape: `(H, W, C)`, `(C, H, W)`, or `(B, C, H, W)`. |
| `img2` | `torch.Tensor` | Base layer returned where the mask is zero. Must match `img1`'s shape and dtype. |
| `mask` | `torch.Tensor` | Spatial mask. BCHW inputs accept `(H, W)`, `(B, H, W)`, or `(B, 1, H, W)`. |
| `stream` | `Optional[torch.cuda.Stream]` | If `None`, CUDA execution is synchronous. If provided, execution is asynchronous on that exact stream. Ignored for CPU tensors. |
| `layout` | `Optional[str]` | Explicitly selects `HWC`, `CHW`, `BCHW`, or `NCHW`. Usually inferred from image and mask shapes. |
| `mode` | `str` | Blend operation: `linear`, `normal`, `multiply`, `screen`, or `overlay`. |

**Returns:** A contiguous tensor with shape, dtype, and device matching `img1`.
Empty spatial dimensions return an empty tensor.

### Input Requirements

- All tensors must use the same device and dtype.
- Images must use `HWC`, `CHW`, or `BCHW`/`NCHW` layout.
- Batched masks may be shared across the batch or supplied per sample.
- Supported dtypes are `torch.uint8`, `torch.float16`, and `torch.float32`.
- Non-contiguous tensors are accepted and converted internally.
- Floating-point mask values outside `[0.0, 1.0]` are not clamped.
- Ambiguous 3D shapes prefer `HWC`; pass `layout="CHW"` to override detection.

Invalid shapes, devices, or mixed dtypes raise `ValueError`. Unsupported dtypes
and invalid CUDA stream objects raise `TypeError`.

## Blend Modes

The selected mode is evaluated first and then composited over `img2` using the
mask:

```text
mode_result = apply_mode(img1, img2)
output = mode_result * alpha + img2 * (1 - alpha)
```

This preserves the existing mask behavior:

- A zero mask returns `img2`.
- A fully opaque mask returns the selected mode result.
- `linear` and `normal` preserve the original linear blend behavior.

For `max_val = 255` with `uint8` and `max_val = 1` with floating-point tensors:

```text
linear:
    mode_result = img1

multiply:
    mode_result = img1 * img2 / max_val

screen:
    mode_result = max_val - ((max_val - img1) * (max_val - img2) / max_val)

overlay:
    if img2 <= max_val / 2:
        mode_result = 2 * img1 * img2 / max_val
    else:
        mode_result = max_val - 2 * (max_val - img1) * (max_val - img2) / max_val
```

`overlay` treats `img2` as the base layer when selecting its lower or upper
formula branch.

```python
result = ImageBlender.blend(
    img1,
    img2,
    mask,
    mode="screen",
)
```

## Native Extension Architecture

- `bindings.cpp` exposes the native dispatcher to Python and contains no kernel logic.
- `blend.cpp` validates native inputs, builds shared metadata, and selects the CPU or CUDA backend.
- `blend_cpu.cpp` contains only the CPU implementation.
- `blend_cuda.cu` contains only CUDA stream handling, kernel launch, and CUDA execution.
- `blend_common.h` owns layout metadata and mask-index calculation shared by CPU and CUDA.
- `blend_modes.h` contains compile-time blend mode formulas shared by CPU and CUDA.
- `blend.h` defines the internal extension interface between translation units.

For channel-first tensors, CUDA uses `gridDim.z` rather than `blockDim.z` for
the batch dimension. This preserves the full thread block for contiguous spatial
work, avoids per-element batch/channel division, and keeps NCHW memory accesses
coalesced within each channel plane.

### CUDA Vectorization

The CUDA backend selects vectorized kernels automatically:

- `HWC` uses `uchar4` or `float4` when the image has exactly four channels.
  Each thread loads and stores one complete four-channel pixel using one mask value.
- `CHW` and `BCHW` use four-value vectors across adjacent spatial positions when
  `H * W` is divisible by four.
- `uint8` uses `uchar4`; `float32` uses `float4`.
- All image, mask, and output pointers must satisfy the vector type's alignment.
- RGB `HWC`, `float16`, non-divisible spatial sizes, and unaligned pointers
  automatically use the scalar kernels.

Vectorization does not change the Python API or output layout. It reduces the
number of memory instructions for bandwidth-bound high-resolution workloads.

## License
MIT License
