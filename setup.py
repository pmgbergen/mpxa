from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools import Extension
import subprocess
import os
import shutil
import site

class CustomInstall(build_ext):
    def run(self):
        # Run the default install command
        super().run()
        # Ensure the CMake build is triggered
        build_dir = "build"
        bindings_dir = os.path.join(build_dir, "bindings/python")

        subprocess.check_call(["cmake", "-B", build_dir])
        subprocess.check_call(["cmake", "--build", build_dir, "--parallel"])
        # The build artefact is _mpxa.so

        # Install with -e flag: copy _mpxa.so to src/mpxa
        # Regular install (without -e flag): copy to site-packages
        # In both cases, the C++ binding will be accessed by `mpxa._mpxa`
        if self.editable_mode:
            target_dir = os.path.join("src", "mpxa")
        else:
            target_dir = os.path.join(site.getsitepackages()[0], "mpxa")
        os.makedirs(target_dir, exist_ok=True)

        for file in os.listdir(bindings_dir):
            if file.endswith(".so") and file.startswith("_mpxa"):
                shutil.copy(os.path.join(bindings_dir, file), target_dir)


setup(
    packages=find_packages(where="src"),  # Find packages in src directory
    package_dir={"": "src"},  # Tell setuptools packages are under src/
    package_data={"mpxa": ["*.so", "*.py"]},  # Include .so files and Python files
    install_requires=["numpy", "pybind11>=2.6.0", "scipy", "porepy"],  # Match pyproject.toml
    ext_modules=[Extension("_mpxa", sources=[])],  # Dummy to trigger build_ext
    cmdclass={
        "build_ext": CustomInstall,
    },
)