from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os
import shutil
import site


class CustomInstall(install):
    def run(self):
        # Run the default install command
        super().run()
        # Ensure the CMake build is triggered
        build_dir = "build"
        bindings_dir = os.path.join(build_dir, "bindings/python")
        target_dir = site.getsitepackages()[0]
        # target_dir = os.path.join(os.getcwd(), ".venv/lib/python3.13/site-packages")
        os.makedirs(target_dir, exist_ok=True)

        # Run CMake and make
        subprocess.check_call(["cmake", "-B", build_dir])
        subprocess.check_call(["cmake", "--build", build_dir])

        # Copy the built .so file to the package directory
        for file in os.listdir(bindings_dir):
            if file.endswith(".so") and file.startswith("_mpxa"):
                shutil.copy(os.path.join(bindings_dir, file), target_dir)


setup(
    packages=find_packages(),  # Specify the package containing the bindings
    package_data={"mpxa": ["*.so"]},  # Include the prebuilt grid.so file
    install_requires=["pybind11>=2.6.0"],  # Ensure pybind11 is installed
    cmdclass={
        "install": CustomInstall,
    },
)
