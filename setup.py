from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "grid",
        ["bindings/python/grid_bindings.cpp"],  # Updated path
        include_dirs=["include"],
        extra_compile_args=["-std=c++17"],
    ),
]

setup(
    name="mpxa",
    version="0.1.0",
    author="Eirik Keilegavlen",
    description="Python bindings for the MPXA C++ library",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    install_requires=["pybind11>=2.6.0"],  # Ensure pybind11 is installed
)
