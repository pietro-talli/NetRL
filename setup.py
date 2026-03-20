"""
NetRL setup.py
==============
Builds the pybind11 C++ extensions:
1. netcomm - Gilbert-Elliott channel (pure C++)
2. _netrl_ext - NS3 WiFi channel backend (if NS3 is installed)

Quick start
-----------
# Install with NS3 support (recommended):
    pip install -e .

# This automatically:
    - Installs ns3 via pip
    - Detects NS3 installation
    - Builds netcomm extension
    - Builds _netrl_ext extension (NS3 pybind11 backend)

# Build the .so in-place (importable without install):
    python setup.py build_ext --inplace

# Clean rebuild:
    python setup.py clean --all && pip install -e .

Notes
-----
- Requires GCC >= 9 or Clang >= 10 (C++17/C++20).
- cmake is NOT needed; setuptools + pybind11.setup_helpers handle everything.
- NS3 is now installed automatically via pip (pip install ns3)
- The built .so files will be named something like:
    netcomm.cpython-311-x86_64-linux-gnu.so
    _netrl_ext.cpython-311-x86_64-linux-gnu.so
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import pybind11
import subprocess
import sys
import os


def get_ns3_flags():
    """Detect NS3 installation and return include/lib flags."""
    try:
        # Try to detect pip-installed ns3
        ns3_spec = __import__('importlib.util').util.find_spec('ns3')
        if ns3_spec and ns3_spec.submodule_search_locations:
            for p in list(ns3_spec.submodule_search_locations):
                candidate = os.path.join(os.path.dirname(p), 'ns3')

                # First check: look for simulator.h
                simulator_h = os.path.join(candidate, 'include', 'ns3', 'simulator.h')
                if os.path.isfile(simulator_h):
                    ns3_inc = os.path.join(candidate, 'include')
                    ns3_lib = os.path.join(candidate, 'lib64')

                    if os.path.isdir(ns3_lib):
                        # Get version from library (e.g., libns3.44-core.so)
                        try:
                            import glob
                            libs = glob.glob(os.path.join(ns3_lib, 'libns3.*-core*.so'))
                            if libs:
                                import re
                                match = re.search(r'libns3\.(\d+)', libs[0])
                                if match:
                                    ver = match.group(1)
                                    print(f"[netrl] Detected NS3 version {ver} at {candidate}")
                                    return {
                                        'include_dirs': [ns3_inc],
                                        'library_dirs': [ns3_lib],
                                        'libraries': [
                                            f'ns3.{ver}-core',
                                            f'ns3.{ver}-network',
                                            f'ns3.{ver}-internet',
                                            f'ns3.{ver}-wifi',
                                            f'ns3.{ver}-mobility',
                                            f'ns3.{ver}-propagation',
                                        ],
                                    }
                        except Exception as e:
                            print(f"[netrl] Warning: could not extract NS3 version: {e}")
    except Exception as e:
        print(f"[netrl] Warning: NS3 detection failed: {e}")

    # Return empty dict if NS3 not found (will be optional)
    print("[netrl] NS3 not found - netrl_ext extension will not be built")
    return {'include_dirs': [], 'library_dirs': [], 'libraries': []}


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

# Add NS3 WiFi pybind11 extension if NS3 is available
ns3_flags = get_ns3_flags()
if ns3_flags['libraries']:
    print("[netrl] Building _netrl_ext (NS3 WiFi pybind11 backend)")
    extra_link_args = [f"-Wl,-rpath,{lib_dir}" for lib_dir in ns3_flags['library_dirs']]
    ext_modules.append(
        Pybind11Extension(
            name="_netrl_ext",
            sources=["src/ns3_wifi_channel_pybind11.cpp"],
            include_dirs=[pybind11.get_include()] + ns3_flags['include_dirs'],
            library_dirs=ns3_flags['library_dirs'],
            libraries=ns3_flags['libraries'],
            extra_compile_args=[
                "-O3",
                "-std=c++20",
                "-fPIC",
                "-fvisibility=hidden",
            ],
            extra_link_args=extra_link_args,
            cxx_std=20,
        )
    )
else:
    print("[netrl] NS3 not available - netrl_ext extension will NOT be built")


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
        "ns3>=4.44",  # NS3 simulator (for fast pybind11 channel)
    ],
    extras_require={
        "dev": ["pytest>=7.0", "matplotlib>=3.7"],
    },
)
