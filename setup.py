from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="torch_blend",
    version="0.1.0",
    packages=["torch_blend"],
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
