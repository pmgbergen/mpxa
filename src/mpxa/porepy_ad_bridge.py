from __future__ import annotations

from typing import Callable

import numpy as np
import porepy as pp
from porepy.numerics.ad.ad_utils import MergedOperator, wrap_discretization
from porepy.numerics.ad.discretizations import Discretization

import mpxa
from mpxa import _mpxa


def _rotate_2d_grid_to_xy_plane(g: pp.Grid, k: pp.SecondOrderTensor) -> pp.Grid:
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


def rotate_vector_source_from_xy_plane_to_original(
    sd: pp.Grid,
    vector_source_dim: int,
    vector_source_glob: np.ndarray,
    bound_pressure_vector_source_glob: np.ndarray,
) -> np.ndarray:
    vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
        vector_source_glob, ambient_dim=2
    )
    bound_pressure_vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
        bound_pressure_vector_source_glob, ambient_dim=2
    )
    # vector_source = mpxa.convert_matrix_mpxa_to_scipy(vector_source_glob)
    # bound_pressure_vector_source = mpxa.convert_matrix_mpxa_to_scipy(
    #     bound_pressure_vector_source_glob
    # )

    # By assumption:
    vector_source_dim = 3

    # Use the same mapping of the geometry as was done in
    # self._flux_discretization(). This mapping is deterministic, thus the
    # rotation matrix should be the same as applied before. In this case, we
    # only need the rotation, and the active dimensions
    *_, R, dim, _ = pp.map_geometry.map_grid(sd)

    # We need to pick out the parts of the rotation matrix that gives the
    # in-plane (relative to the grid) parts, and apply this to all cells in the
    # grid. The simplest way to do this is to expand R[dim] via sps.block_diags,
    # however this scales poorly with the number of blocks. Instead, use the
    # existing workaround to create a csr matrix based on R, and then pick out
    # the right parts of that one.
    full_rot_mat = pp.matrix_operations.csr_matrix_from_dense_blocks(
        # Replicate R with the right ordering of data elements
        np.tile(R.ravel(), (1, sd.num_cells)).ravel(),
        # size of the blocks - this will always be the dimension of the rotation
        # matrix, that is, 3 - due to the grid coordinates being 3d. If the
        # ambient dimension really is 2, this is adjusted below.
        3,
        # Number of blocks
        sd.num_cells,
    )
    # Get the right components of the rotation matrix.
    dim_expanded = np.where(dim)[0].reshape((-1, 1)) + vector_source_dim * np.array(
        np.arange(sd.num_cells)
    )
    # Dump the irrelevant rows of the global rotation matrix.
    glob_R = full_rot_mat[dim_expanded.ravel("F")]
    # Append a mapping from the ambient dimension onto the plane of this grid
    vector_source *= glob_R
    bound_pressure_vector_source *= glob_R
    return vector_source, bound_pressure_vector_source


def _extract_rotate_vector_source(sd, ambient_dim, discr_cpp):
    vector_source_cpp = discr_cpp.vector_source
    bound_pressure_vector_source_cpp = discr_cpp.bound_pressure_vector_source

    if sd.dim == 3:
        assert ambient_dim == 3, (
            "If the grid is 3d, the ambient dimension must be 3 as well."
        )
        # No need for dimension reduction or rotation.
        vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
            vector_source_cpp, ambient_dim=ambient_dim
        )
        bound_pressure_vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
            bound_pressure_vector_source_cpp, ambient_dim=ambient_dim
        )
    elif sd.dim == 2 and ambient_dim == 2:
        # The C++ code has discretized the vector source as if it is 3d. Use a
        # special conversion method which strips away the z-coordinate from
        # column space of the vector source matrix and its boundary terms.
        vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
            vector_source_cpp, ambient_dim=ambient_dim
        )
        bound_pressure_vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
            bound_pressure_vector_source_cpp, ambient_dim=ambient_dim
        )
    elif sd.dim == 2 and ambient_dim == 3:
        # The C++ discretization have the right number of columns, but they were
        # discretized in the xy-plane. The simplest option is to right-multiply
        # the discretizations with a rotation to that xy-plane, as is done in the
        # called function.
        vector_source, bound_pressure_vector_source = (
            rotate_vector_source_from_xy_plane_to_original(
                sd,
                ambient_dim,
                vector_source_cpp,
                bound_pressure_vector_source_cpp,
            )
        )
    else:  # This is a 1d grid. When we introduce a 1d grid aligned with the y-axis
        # this will need an extension (possibly outsourcing to tpfa), but keep the
        # code for now, to make the tests formally pass.
        vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
            vector_source_cpp, ambient_dim=ambient_dim
        )
        bound_pressure_vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
            bound_pressure_vector_source_cpp, ambient_dim=ambient_dim
        )
    return vector_source, bound_pressure_vector_source


class Tpfa(pp.FVElliptic):
    def __init__(self, keyword: str) -> None:
        super().__init__(keyword)

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        K_pp = parameter_dictionary["second_order_tensor"]
        bc_pp = parameter_dictionary["bc"]
        g_pp = sd

        if g_pp.dim == 2:
            g_pp, K_pp = _rotate_2d_grid_to_xy_plane(g_pp, K_pp)

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

        vector_source, bound_pressure_vector_source = _extract_rotate_vector_source(
            sd, ambient_dim, tpfa_cpp
        )

        data[pp.DISCRETIZATION_MATRICES][self.keyword] = {
            key: mpxa.convert_matrix_mpxa_to_scipy(value)
            for key, value in {
                "flux": tpfa_cpp.flux,
                "bound_flux": tpfa_cpp.bound_flux,
                "bound_pressure_face": tpfa_cpp.bound_pressure_face,
                "bound_pressure_cell": tpfa_cpp.bound_pressure_cell,
            }.items()
        } | {
            "vector_source": vector_source,
            "bound_pressure_vector_source": bound_pressure_vector_source,
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

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        K_pp = parameter_dictionary["second_order_tensor"]
        bc_pp = parameter_dictionary["bc"]
        if sd.dim == 2:
            g_pp, K_pp = _rotate_2d_grid_to_xy_plane(sd, K_pp)
        else:
            g_pp = sd

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

        vector_source, bound_pressure_vector_source = _extract_rotate_vector_source(
            sd, ambient_dim, mpfa_cpp
        )

        data[pp.DISCRETIZATION_MATRICES][self.keyword] = {
            key: mpxa.convert_matrix_mpxa_to_scipy(value)
            for key, value in {
                "flux": mpfa_cpp.flux,
                "bound_flux": mpfa_cpp.bound_flux,
                "bound_pressure_face": mpfa_cpp.bound_pressure_face,
                "bound_pressure_cell": mpfa_cpp.bound_pressure_cell,
            }.items()
        } | {
            "vector_source": vector_source,
            "bound_pressure_vector_source": bound_pressure_vector_source,
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
