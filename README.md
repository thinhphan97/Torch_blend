# Torch Blend Package

A high-performance PyTorch C++/CUDA extension for blending two images using a mask. 
This package provides a custom PyTorch operator that runs natively on CUDA (with a CPU fallback), supporting both **synchronous** and **asynchronous** execution via custom CUDA streams.

## ✨ Features

- **Multi-Dtype Support**: Supports `uint8`, `float32`, and `float16` tensors on CPU and CUDA. Other dtypes are rejected explicitly.
- **Contiguous Native Operations**: Directly accesses PyTorch tensor memory for contiguous inputs. Non-contiguous inputs are converted automatically before dispatch.
- **Auto Sync/Async**: Runs synchronously if no stream is provided and asynchronously on the exact `torch.cuda.Stream` passed by the caller.
- **Device-Aware**: Falls back to a pure C++ CPU loop if tensors are on CPU, with friendly warnings if streams are misused.
- **Flexible Image Layout**: Supports HWC images with one or more channels, including grayscale, RGB, and RGBA.
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
│       └── blend_cuda.cu    # C++/CUDA kernel and Pybind11 bindings
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
empty inputs, output metadata, and input validation.

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

### 2. Deep Learning Pipeline (float32 / float16)
Floating-point images and masks are interpreted directly; they are not normalized
by the package. For standard alpha blending, use values in `[0.0, 1.0]`.
Mask values outside this range are allowed and perform linear extrapolation.

```python
import torch
from torch_blend import ImageBlender

# Assume img1 and img2 have shape (H, W, C).
# The mask must have shape (H, W).

# Asynchronous Blend on GPU using float16 (half precision) for speed
my_stream = torch.cuda.Stream()

# The stream is used directly, even outside a stream context manager.
result_async = ImageBlender.blend(
    img1.half().cuda(),
    img2.half().cuda(),
    mask.half().cuda(),
    stream=my_stream,
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

### `ImageBlender.blend(img1, img2, mask, stream=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `img1` | `torch.Tensor` | Background image. Shape: `(H, W, C)`. Dtype: `uint8`, `float32`, or `float16`. |
| `img2` | `torch.Tensor` | Foreground image. Must match `img1`'s shape and dtype. |
| `mask` | `torch.Tensor` | Mask with shape `(H, W)` and the same dtype as the images. Full opacity is `255` for `uint8` and `1.0` for floating-point tensors. |
| `stream` | `Optional[torch.cuda.Stream]` | If `None`, CUDA execution is synchronous. If provided, execution is asynchronous on that exact stream. Ignored for CPU tensors. |

**Returns:** A contiguous tensor with shape, dtype, and device matching `img1`.
Empty spatial dimensions return an empty tensor.

### Input Requirements

- All tensors must use the same device and dtype.
- Images must have shape `(H, W, C)`; masks must have shape `(H, W)`.
- Supported dtypes are `torch.uint8`, `torch.float16`, and `torch.float32`.
- Non-contiguous tensors are accepted and converted internally.
- Floating-point mask values outside `[0.0, 1.0]` are not clamped.

Invalid shapes, devices, or mixed dtypes raise `ValueError`. Unsupported dtypes
and invalid CUDA stream objects raise `TypeError`.

## License
MIT License
