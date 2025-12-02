from __future__ import annotations

import porepy as pp

from . import porepy_bridge  # Import the bridge functions module


class MpfaNonAd(pp.Mpfa):

    def __init__(self, keyword: str) -> None:
        super(MpfaNonAd, self).__init__(keyword)

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        
        K_pp = data[pp.PARAMETERS][self.keyword]["second_order_tensor"]
        bc_pp = data[pp.PARAMETERS][self.keyword]["bc"]
        g_pp = sd

        K_cpp = porepy_bridge.convert_tensor(K_pp, g_pp.dim)
        bc_cpp = porepy_bridge.convert_bc(bc_pp)
        g_cpp = porepy_bridge.convert_grid(g_pp)

        # Import _mpxa directly to avoid circular import
        from _mpxa import mpfa
        mpfa_cpp = mpfa(g_cpp, K_cpp, bc_cpp)

        # Prepare the cpp discretization
        for attribute in [
            "vector_source",
            "flux",
            "bound_flux",
            "bound_pressure_face",
            "bound_pressure_cell",
            "bound_pressure_vector_source",
        ]:
            data[pp.DISCRETIZATION_MATRICES][self.keyword][attribute] = porepy_bridge.convert_matrix(getattr(mpfa_cpp, attribute))
