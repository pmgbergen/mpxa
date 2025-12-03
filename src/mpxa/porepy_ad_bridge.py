from __future__ import annotations

from typing import Callable
import porepy as pp
from porepy.numerics.ad._ad_utils import MergedOperator, wrap_discretization
import mpxa

class Tpfa(pp.Tpfa):

    def __init__(self, keyword: str) -> None:
        super(Tpfa, self).__init__(keyword)

    def discretize(self, sd: pp.Grid, data: dict) -> None:

        K_pp = data[pp.PARAMETERS][self.keyword]["second_order_tensor"]
        bc_pp = data[pp.PARAMETERS][self.keyword]["bc"]
        g_pp = sd

        K_cpp = mpxa.convert_tensor(K_pp, g_pp.dim)
        bc_cpp = mpxa.convert_bc(bc_pp)
        g_cpp = mpxa.convert_grid(g_pp)

        # Import _mpxa directly to avoid circular import
        #from _mpxa import tpfa
        tpfa_cpp = mpxa.tpfa(g_cpp, K_cpp, bc_cpp)

        # Prepare the cpp discretization
        for attribute in [
            "vector_source",
            "flux",
            "bound_flux",
            "bound_pressure_face",
            "bound_pressure_cell",
            "bound_pressure_vector_source",
        ]:
            data[pp.DISCRETIZATION_MATRICES][self.keyword][attribute] = mpxa.convert_matrix(getattr(tpfa_cpp, attribute))

class Mpfa(pp.Mpfa):

    def __init__(self, keyword: str) -> None:
        super(Mpfa, self).__init__(keyword)

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        
        K_pp = data[pp.PARAMETERS][self.keyword]["second_order_tensor"]
        bc_pp = data[pp.PARAMETERS][self.keyword]["bc"]
        g_pp = sd

        K_cpp = mpxa.convert_tensor(K_pp, g_pp.dim)
        bc_cpp = mpxa.convert_bc(bc_pp)
        g_cpp = mpxa.convert_grid(g_pp)

        # Import _mpxa directly to avoid circular import
        #from _mpxa import mpfa
        mpfa_cpp = mpxa.mpfa(g_cpp, K_cpp, bc_cpp)

        # Prepare the cpp discretization
        for attribute in [
            "vector_source",
            "flux",
            "bound_flux",
            "bound_pressure_face",
            "bound_pressure_cell",
            "bound_pressure_vector_source",
        ]:
            data[pp.DISCRETIZATION_MATRICES][self.keyword][attribute] = mpxa.convert_matrix(getattr(mpfa_cpp, attribute))


class TpfaAd(pp.Discretization):

    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._name = "Tpfa"
        self._discretization = Tpfa(keyword)
        self.keyword = keyword

        # Prepare the cpp discretization
        self.flux: Callable[[], MergedOperator]
        self.bound_flux: Callable[[], MergedOperator]
        self.bound_pressure_cell: Callable[[], MergedOperator]
        self.bound_pressure_face: Callable[[], MergedOperator]
        self.vector_source: Callable[[], MergedOperator]
        self.bound_pressure_vector_source: Callable[[], MergedOperator]

        wrap_discretization(self, self._discretization, subdomains=subdomains)

class MpfaAd(pp.Discretization):

    def __init__(self, keyword: str, subdomains: list[pp.Grid]) -> None:
        self.subdomains = subdomains
        self._name = "Mpfa"
        self._discretization = Mpfa(keyword)
        self.keyword = keyword

        # Prepare the cpp discretization
        self.flux: Callable[[], MergedOperator]
        self.bound_flux: Callable[[], MergedOperator]
        self.bound_pressure_cell: Callable[[], MergedOperator]
        self.bound_pressure_face: Callable[[], MergedOperator]
        self.vector_source: Callable[[], MergedOperator]
        self.bound_pressure_vector_source: Callable[[], MergedOperator]

        wrap_discretization(self, self._discretization, subdomains=subdomains)
