from __future__ import annotations

from typing import Callable

import scipy.sparse as sps
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
        dim,
        nodes,
    ) = pp.map_geometry.map_grid(g)

    for s, field in zip(
        ["cell_centers", "face_normals", "face_centers", "nodes"],
        [cell_centers, face_normals, face_centers, nodes],
    ):
        grid_field = getattr(g, s)
        grid_field[: g.dim] = field[: g.dim]
        grid_field[g.dim :] = 0
        setattr(g, s, grid_field)

    # Rotate the permeability tensor and delete last dimension
    k.values = np.tensordot(R.T, np.tensordot(R, k.values, (1, 0)), (0, 1))
    k.values = np.delete(k.values, (2), axis=0)
    k.values = np.delete(k.values, (2), axis=1)
    return g, k, (R, dim)


def _rotate_1d_grid_to_x_axis(g: pp.Grid, k: pp.SecondOrderTensor) -> pp.Grid:
    # Unit vector along the grid. This breaks if the grid is not a straight line.
    g = g.copy()
    v = g.nodes[:, 1] - g.nodes[:, 0]
    v /= np.linalg.norm(v)

    # Project the grid geometry onto the line defined by v, and delete the other
    # dimensions.
    for s in ["cell_centers", "face_normals", "face_centers", "nodes"]:
        grid_field = getattr(g, s)
        grid_field[0] = np.einsum("i,ij->j", v, grid_field)
        grid_field[1:] = 0
        setattr(g, s, grid_field)

    # Project the permeability tensor onto the line defined by v.
    k_principal = np.einsum("i,ijk,j->k", v, k.values, v)
    k.values = np.zeros((3, 3, g.num_cells))
    k.values[0, 0] = k_principal
    return g, k, v


def rotate_vector_source_from_xy_plane_to_original(
    sd: pp.Grid,
    vector_source_dim: int,
    vector_source_glob: np.ndarray,
    bound_pressure_vector_source_glob: np.ndarray,
    rot_info,
) -> np.ndarray:
    # Zero out non-active dimensions.
    vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
        vector_source_glob, ambient_dim=sd.dim
    )
    bound_pressure_vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
        bound_pressure_vector_source_glob, ambient_dim=sd.dim
    )
    R, active_dim = rot_info

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
    dim_expanded = np.where(active_dim)[0].reshape(
        (-1, 1)
    ) + vector_source_dim * np.array(np.arange(sd.num_cells))
    # Dump the irrelevant rows of the global rotation matrix.
    glob_R = full_rot_mat[dim_expanded.ravel("F")]
    # Append a mapping from the ambient dimension onto the plane of this grid
    vector_source *= glob_R
    bound_pressure_vector_source *= glob_R
    return vector_source, bound_pressure_vector_source


def rotate_vector_source_from_1d_to_original(
    sd: pp.Grid,
    vector_source_dim: int,
    vector_source_glob: np.ndarray,
    bound_pressure_vector_source_glob: np.ndarray,
    rot_info: np.ndarray,
):
    vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
        vector_source_glob, ambient_dim=sd.dim
    )
    bound_pressure_vector_source = mpxa.convert_vector_source_mpxa_to_scipy(
        bound_pressure_vector_source_glob, ambient_dim=sd.dim
    )

    # Pick out the part of the vector defining the direction of the grid that lies in
    # the ambient dimensions.
    v = rot_info[:vector_source_dim]
    data = np.tile(v, sd.num_cells)
    indices = np.arange(sd.num_cells * vector_source_dim)
    indptr = np.arange(0, sd.num_cells * vector_source_dim + 1, vector_source_dim)
    rot_mat = sps.csr_matrix(
        (data, indices, indptr), shape=(sd.num_cells, sd.num_cells * vector_source_dim)
    )
    vector_source = vector_source @ rot_mat
    bound_pressure_vector_source = bound_pressure_vector_source @ rot_mat
    return vector_source, bound_pressure_vector_source


def _extract_rotate_vector_source(sd, ambient_dim, discr_cpp, rot_info):
    vector_source_cpp = discr_cpp.vector_source
    bound_pressure_vector_source_cpp = discr_cpp.bound_pressure_vector_source

    if sd.dim == 3 or sd.dim == 0:
        assert ambient_dim >= sd.dim, (
            "The ambient dimension cannot be smaller than the grid dimension."
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
                rot_info,
            )
        )
    else:
        # This is a 1d grid. When we introduce a 1d grid aligned with the y-axis
        # this will need an extension (possibly outsourcing to tpfa), but keep the
        # code for now, to make the tests formally pass.
        vector_source, bound_pressure_vector_source = (
            rotate_vector_source_from_1d_to_original(
                sd,
                ambient_dim,
                vector_source_cpp,
                bound_pressure_vector_source_cpp,
                rot_info,
            )
        )
    return vector_source, bound_pressure_vector_source


def _store_discretization_matrices(sd, data, keyword, discr_cpp, rot_info):

    # Convert the discretization matrices to scipy format.
    try:
        ambient_dim = data[pp.PARAMETERS][keyword]["ambient_dimension"]
    except KeyError as e:
        raise ValueError(
            "ambient_dimension must be provided in parameter dictionary for the "
            "C++ discretization"
        ) from e

    # Special treatment for vector source terms.
    vector_source, bound_pressure_vector_source = _extract_rotate_vector_source(
        sd, ambient_dim, discr_cpp, rot_info
    )
    data[pp.DISCRETIZATION_MATRICES][keyword] = {
        key: mpxa.convert_matrix_mpxa_to_scipy(value)
        for key, value in {
            "flux": discr_cpp.flux,
            "bound_flux": discr_cpp.bound_flux,
            "bound_pressure_face": discr_cpp.bound_pressure_face,
            "bound_pressure_cell": discr_cpp.bound_pressure_cell,
        }.items()
    } | {
        "vector_source": vector_source,
        "bound_pressure_vector_source": bound_pressure_vector_source,
    }


def _extract_data_for_discretization(sd, parameter_dictionary):
    K_pp = parameter_dictionary["second_order_tensor"]
    bc_pp = parameter_dictionary["bc"]
    # Rotate the grid and permeability tensor.
    if sd.dim == 2:
        g_pp, K_pp, rot_info = _rotate_2d_grid_to_xy_plane(sd, K_pp)
    elif sd.dim == 1:
        g_pp, K_pp, rot_info = _rotate_1d_grid_to_x_axis(sd, K_pp)
    else:
        g_pp = sd
        rot_info = None

    K_cpp = mpxa.convert_tensor_to_mpxa(K_pp, g_pp.dim)
    bc_cpp = mpxa.convert_bc_to_mpxa(bc_pp)
    g_cpp = mpxa.convert_grid_to_mpxa(g_pp)

    return g_cpp, K_cpp, bc_cpp, rot_info


class Tpfa(pp.FVElliptic):
    def __init__(self, keyword: str) -> None:
        super().__init__(keyword)

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        g_cpp, K_cpp, bc_cpp, rot_info = _extract_data_for_discretization(
            sd, data[pp.PARAMETERS][self.keyword]
        )

        tpfa_cpp = _mpxa.tpfa(g_cpp, K_cpp, bc_cpp)
        _store_discretization_matrices(sd, data, self.keyword, tpfa_cpp, rot_info)


class Mpfa(pp.FVElliptic):
    def __init__(self, keyword: str) -> None:
        super(Mpfa, self).__init__(keyword)

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        g_cpp, K_cpp, bc_cpp, rot_info = _extract_data_for_discretization(
            sd, data[pp.PARAMETERS][self.keyword]
        )

        mpfa_cpp = _mpxa.mpfa(g_cpp, K_cpp, bc_cpp)
        _store_discretization_matrices(sd, data, self.keyword, mpfa_cpp, rot_info)


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
