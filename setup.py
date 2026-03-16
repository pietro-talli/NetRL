"""
NetRL setup.py
==============
Builds the pybind11 C++ extension (netcomm) and installs the netrl package.

Quick start
-----------
# Install Python deps + build C++ extension + install package in editable mode:
    pip install pybind11 numpy gymnasium
    pip install -e .

# Build the .so in-place (importable without install):
    python setup.py build_ext --inplace

# Clean rebuild:
    python setup.py clean --all && pip install -e .

Notes
-----
- Requires GCC >= 9 or Clang >= 10 (C++17).
- cmake is NOT needed; setuptools + pybind11.setup_helpers handle everything.
- The built .so will be named something like:
    netcomm.cpython-311-x86_64-linux-gnu.so
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import pybind11

ext_modules = [
    Pybind11Extension(
        name="netcomm",
        sources=["src/netcomm.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=[
            "-O3",
            "-std=c++17",
            "-fvisibility=hidden",  # reduces .so size
        ],
        cxx_std=17,
    ),
]

setup(
    name="netrl",
    version="0.1.0",
    description=(
        "Networked RL simulation platform — "
        "gymnasium wrapper with Gilbert-Elliott channel (C++ backend)"
    ),
    packages=find_packages(exclude=["tests*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.24",
        "pybind11>=2.11",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "matplotlib>=3.7"],
    },
)
