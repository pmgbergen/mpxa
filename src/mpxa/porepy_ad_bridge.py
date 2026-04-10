from __future__ import annotations

from typing import Callable

import numpy as np
import porepy as pp
from porepy.numerics.ad.ad_utils import MergedOperator, wrap_discretization
from porepy.numerics.ad.discretizations import Discretization

import mpxa
from mpxa import _mpxa


class Tpfa(pp.FVElliptic):
    def __init__(self, keyword: str) -> None:
        super().__init__(keyword)

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        K_pp = parameter_dictionary["second_order_tensor"]
        bc_pp = parameter_dictionary["bc"]
        g_pp = sd

        K_cpp = mpxa.convert_tensor_to_mpxa(K_pp, g_pp.dim)
        bc_cpp = mpxa.convert_bc_to_mpxa(bc_pp)
        g_cpp = mpxa.convert_grid_to_mpxa(g_pp)

        tpfa_cpp = _mpxa.tpfa(g_cpp, K_cpp, bc_cpp)

        # Convert the discretization matrices to scipy format.
        try:
            ambient_dim = parameter_dictionary["ambient_dimension"]
        except KeyError as e:
            raise ValueError(
                "ambient_dimension must be provided in parameter dictionary for the "
                f"Tpfa C++ discretization with keyword = {self.keyword}."
            ) from e
        data[pp.DISCRETIZATION_MATRICES][self.keyword] = {
            key: mpxa.convert_matrix_mpxa_to_scipy(value)
            for key, value in {
                "flux": tpfa_cpp.flux,
                "bound_flux": tpfa_cpp.bound_flux,
                "bound_pressure_face": tpfa_cpp.bound_pressure_face,
                "bound_pressure_cell": tpfa_cpp.bound_pressure_cell,
            }.items()
        } | {
            key: mpxa.convert_vector_source_mpxa_to_scipy(
                value, ambient_dim=ambient_dim
            )
            for key, value in {
                "vector_source": tpfa_cpp.vector_source,
                "bound_pressure_vector_source": tpfa_cpp.bound_pressure_vector_source,
            }.items()
        }
        # YZ: I used the lines below to debug tests. These should be deleted when the
        # all tests pass.
        for k, v in data[pp.DISCRETIZATION_MATRICES][self.keyword].items():
            if np.any(np.isnan(v.data)):
                print(k, v.data)
                assert False


class Mpfa(pp.FVElliptic):
    def __init__(self, keyword: str) -> None:
        super(Mpfa, self).__init__(keyword)

    def _rotate_2d_grid_to_xy_plane(
        self, g: pp.Grid, k: pp.SecondOrderTensor
    ) -> pp.Grid:
        # Rotate the grid into the xy plane and delete third dimension. First make a
        # copy to avoid alterations to the input grid
        g = g.copy()
        (
            cell_centers,
            face_normals,
            face_centers,
            R,
            _,
            nodes,
        ) = pp.map_geometry.map_grid(g)
        g.cell_centers[:2] = cell_centers
        g.face_normals[:2] = face_normals
        g.face_centers[:2] = face_centers
        g.nodes[:2] = nodes
        g.cell_centers[2] = 0
        g.face_normals[2] = 0
        g.face_centers[2] = 0
        g.nodes[2] = 0

        # Rotate the permeability tensor and delete last dimension
        k = k.copy()
        k.values = np.tensordot(R.T, np.tensordot(R, k.values, (1, 0)), (0, 1))
        k.values = np.delete(k.values, (2), axis=0)
        k.values = np.delete(k.values, (2), axis=1)
        return g, k

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        K_pp = parameter_dictionary["second_order_tensor"]
        bc_pp = parameter_dictionary["bc"]
        g_pp = sd
        if g_pp.dim == 2:
            g_pp, K_pp = self._rotate_2d_grid_to_xy_plane(g_pp, K_pp)

        K_cpp = mpxa.convert_tensor_to_mpxa(K_pp, g_pp.dim)
        bc_cpp = mpxa.convert_bc_to_mpxa(bc_pp)
        g_cpp = mpxa.convert_grid_to_mpxa(g_pp)

        mpfa_cpp = _mpxa.mpfa(g_cpp, K_cpp, bc_cpp)

        # Convert the discretization matrices to scipy format.
        try:
            ambient_dim = parameter_dictionary["ambient_dimension"]
        except KeyError as e:
            raise ValueError(
                "ambient_dimension must be provided in parameter dictionary for the "
                f"Mpfa C++ discretization with keyword = {self.keyword}."
            ) from e
        data[pp.DISCRETIZATION_MATRICES][self.keyword] = {
            key: mpxa.convert_matrix_mpxa_to_scipy(value)
            for key, value in {
                "flux": mpfa_cpp.flux,
                "bound_flux": mpfa_cpp.bound_flux,
                "bound_pressure_face": mpfa_cpp.bound_pressure_face,
                "bound_pressure_cell": mpfa_cpp.bound_pressure_cell,
            }.items()
        } | {
            key: mpxa.convert_vector_source_mpxa_to_scipy(
                value, ambient_dim=ambient_dim
            )
            for key, value in {
                "vector_source": mpfa_cpp.vector_source,
                "bound_pressure_vector_source": mpfa_cpp.bound_pressure_vector_source,
            }.items()
        }
        # YZ: I used the lines below to debug tests. These should be deleted when the
        # all tests pass.
        for k, v in data[pp.DISCRETIZATION_MATRICES][self.keyword].items():
            if np.any(np.isnan(v.data)):
                print(k, v.data)
                assert False


class TpfaAd(Discretization):
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


class MpfaAd(Discretization):
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
