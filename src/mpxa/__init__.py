# Import all C++ bindings from the compiled module
try:
    from _mpxa import *
except ImportError as e:
    raise ImportError(f"Could not import _mpxa C++ bindings. Make sure the package is properly installed: {e}")

# Import the porepy bridge functions
from .porepy_bridge import convert_matrix, convert_tensor, convert_bc, convert_grid

# Import ad modules
from .porepy_ad_bridge import Tpfa, Mpfa