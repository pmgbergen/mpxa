# Check that the C++ binding is present
try:
    from . import _mpxa
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
