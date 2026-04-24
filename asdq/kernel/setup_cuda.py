#!/usr/bin/env python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="asdq_cuda",
    ext_modules=[
        CUDAExtension(
            name="asdq_cuda",
            sources=["asdq_cuda.cpp", "dequant.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
