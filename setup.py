#!/usr/bin/env python
from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="asdq",
    version="0.1.0",
    description="ASDQ: ASD Quantization for multimodal large models",
    author="ASDQ",
    packages=setuptools.find_packages(),
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
