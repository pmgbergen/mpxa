from setuptools import setup

setup(
    name="mpxa",
    version="0.1.0",
    author="Eirik Keilegavlen",
    description="Python bindings for the MPXA C++ library",
    packages=["bindings.python"],  # Specify the package containing the bindings
    package_data={"bindings.python": ["grid.so"]},  # Include the prebuilt grid.so file
    zip_safe=False,
    install_requires=["pybind11>=2.6.0"],  # Ensure pybind11 is installed
)
