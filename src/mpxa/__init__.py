from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_cpp_bindings():
    package_dir = Path(__file__).resolve().parent
    package_binding = package_dir / "_mpxa.so"
    build_binding = package_dir.parents[1] / "build" / "bindings" / "python" / "_mpxa.so"

    binding_path = package_binding
    if build_binding.exists() and (
        not package_binding.exists()
        or build_binding.stat().st_mtime >= package_binding.stat().st_mtime
    ):
        binding_path = build_binding

    if binding_path == package_binding:
        from . import _mpxa as module

        return module

    spec = spec_from_file_location(f"{__name__}._mpxa", binding_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load _mpxa C++ bindings from {binding_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Check that the C++ binding is present
try:
    _mpxa = _load_cpp_bindings()
except ImportError as e:
    raise ImportError(
        f"Could not import _mpxa C++ bindings. Make sure the package is properly installed: {e}"
    )

# Import the porepy bridge functions
from .porepy_bridge import (
    convert_matrix_mpxa_to_scipy,
    convert_vector_source_mpxa_to_scipy,
    convert_matrix_scipy_to_mpxa,
    convert_tensor_to_mpxa,
    convert_bc_to_mpxa,
    convert_grid_to_mpxa,
)

# Import ad modules
from .porepy_ad_bridge import Tpfa, Mpfa, TpfaAd, MpfaAd
