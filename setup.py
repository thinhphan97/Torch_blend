from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="torch_blend",
    version="0.1.0",
    description="PyTorch CPU/CUDA image blending extension",
    python_requires=">=3.10",
    packages=find_packages(),
    package_data={
        "torch_blend": ["ext/*.cpp", "ext/*.cu", "ext/*.h"],
    },
    include_package_data=True,
    zip_safe=False,
    extras_require={
        "benchmark": [
            "numpy",
            "opencv-python-headless",
        ],
    },
    ext_modules=[
        CUDAExtension(
            name="torch_blend._torch_blend_cuda",
            sources=[
                "torch_blend/ext/bindings.cpp",
                "torch_blend/ext/blend.cpp",
                "torch_blend/ext/blend_cpu.cpp",
                "torch_blend/ext/blend_cuda.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
