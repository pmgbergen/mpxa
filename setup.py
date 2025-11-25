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
    packages=find_packages(where="src"),  # Find packages in src directory
    package_dir={"": "src"},  # Tell setuptools packages are under src/
    package_data={"mpxa": ["*.so", "*.py"]},  # Include .so files and Python files
    install_requires=["numpy", "pybind11>=2.6.0", "scipy", "porepy"],  # Match pyproject.toml
    cmdclass={
        "install": CustomInstall,
    },
)
