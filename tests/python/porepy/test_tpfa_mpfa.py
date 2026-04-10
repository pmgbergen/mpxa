"""Tests for the finite volume discretizations in PorePy against the mpxa bindings.

The test assumes that PorePy is installed.

"""

from typing import Callable

import porepy as pp
import numpy as np
import pytest
import mpxa

import scipy.sparse as sps


g_1d_along_y_axis = pp.CartGrid([2])
g_1d_along_y_axis.nodes[0] = 0
g_1d_along_y_axis.nodes[1] = np.arange(3)

g_1d_along_xyz_diagonal = pp.CartGrid([2])
g_1d_along_xyz_diagonal.nodes[1] = np.arange(3)
g_1d_along_xyz_diagonal.nodes[2] = 2 * np.arange(3)


g_2d_in_yz_plane = pp.CartGrid([2, 2])
g_2d_in_yz_plane.nodes[2] = g_2d_in_yz_plane.nodes[0].copy()
g_2d_in_yz_plane.nodes[0] = 0


@pytest.fixture(
    scope="module",
    params=[
        pp.CartGrid(np.array([3, 3])),  # 2D cartesian grid.
        pp.CartGrid(np.array([3, 3, 3])),  # 3D cartesian grid.
        pp.StructuredTriangleGrid(np.array([3, 3])),  # 2D simplex grid.
        pp.StructuredTetrahedralGrid(np.array([3, 3, 3])),  # 3D simplex grid.
        pp.TensorGrid(x=np.linspace(0, 3, num=10, endpoint=True)),  # 1D line grid.
        pp.PointGrid(pt=np.array([0.5, 0.5, 0.5])),  # 0D point grid.
        g_2d_in_yz_plane,
        g_1d_along_y_axis,
        g_1d_along_xyz_diagonal,
    ],
)
def grid_pp(request) -> pp.Grid:
    grid = request.param
    grid.compute_geometry()
    return grid


def isotropic_tensor(g: pp.Grid):
    num_cells = g.num_cells
    e = np.ones(num_cells)
    return pp.SecondOrderTensor(e)


def anisotropic_tensor(g: pp.Grid):
    num_cells = g.num_cells
    e = np.ones(num_cells)
    return pp.SecondOrderTensor(e, 2 * e, 3 * e)


def full_tensor(g: pp.Grid):
    num_cells = g.num_cells
    e = np.ones(num_cells)
    return pp.SecondOrderTensor(e, 2 * e, 3 * e, 0.1 * e, 0.2 * e, 0.3 * e)


def _compare_matrices(m_0: sps.csr_matrix, m_1: sps.csr_matrix):
    """Compare two matrices for equality."""

    assert m_0.shape == m_1.shape, f"Shape mismatch: {m_0.shape} vs {m_1.shape}"
    diff = m_0 - m_1
    np.testing.assert_allclose(diff.data, 0, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize(
    "tensor_func",
    [
        isotropic_tensor,
        anisotropic_tensor,
        full_tensor,
    ],
)
@pytest.mark.parametrize("discr_type", ["tpfa", "mpfa"])
# Note that the C++ code always treats the ambient dimension as 3 (even for a 2D / 1D
# grid). We test that the code works consistently, regardless of the ambient dimension.
@pytest.mark.parametrize("ambient_dimension", [2, 3])
def test_tpfa_mpfa_discretization(
    grid_pp: pp.Grid,
    tensor_func: Callable[[pp.Grid], pp.SecondOrderTensor],
    discr_type: str,
    ambient_dimension: int,
):
    if ambient_dimension == 2 and grid_pp.dim == 3:
        # ambient_dimension == 2 and grid dimension == 3 does not make sense.
        pytest.skip("Unrealistic combination of parameters")
    elif ambient_dimension == 2 and np.any(grid_pp.nodes[2] > 1e-3):
        pytest.skip("The grid is embedded in 3d, so the ambient dimension cannot be 2.")

    K_pp = tensor_func(grid_pp)
    bc_pp = pp.BoundaryCondition(grid_pp)
    # Setting non-trivial boundary conditions for grids with dim > 0.

    # Observation: something is wrong in treatment of Dirichlet bc for tpfa. It happens
    # with anisotropic and general tensor (isotropic is ok).
    if grid_pp.dim > 0:
        # Each second boundary face is Dirichlet, others - Neumann.
        boundary_faces = grid_pp.get_all_boundary_faces()
        bc_pp.is_dir[boundary_faces[::2]] = True
        bc_pp.is_neu[boundary_faces[::2]] = False

    key = "flow"

    if discr_type == "tpfa":
        discr_pp = pp.Tpfa(key)
    elif discr_type == "mpfa":
        discr_pp = pp.Mpfa(key)

    # Set the parameters for the discretization.
    data_pp = {
        pp.PARAMETERS: {
            key: {
                "second_order_tensor": K_pp,
                "bc": bc_pp,
                "ambient_dimension": ambient_dimension,
            }
        },
        pp.DISCRETIZATION_MATRICES: {key: {}},
    }
    discr_pp.discretize(grid_pp, data_pp)

    if discr_type == "tpfa":
        discr_cpp = mpxa.Tpfa(key)
    elif discr_type == "mpfa":
        discr_cpp = mpxa.Mpfa(key)
    data_cpp = {
        pp.PARAMETERS: {
            key: {
                "second_order_tensor": K_pp,
                "bc": bc_pp,
                "ambient_dimension": ambient_dimension,
            }
        },
        pp.DISCRETIZATION_MATRICES: {key: {}},
    }

    discr_cpp.discretize(grid_pp, data_cpp)

    for attribute in [
        "flux",
        "bound_flux",
        "bound_pressure_face",
        "bound_pressure_cell",
        "vector_source",
        "bound_pressure_vector_source",
    ]:
        m_pp = data_pp[pp.DISCRETIZATION_MATRICES][key][attribute]
        m_cpp = data_cpp[pp.DISCRETIZATION_MATRICES][key][attribute]

        # The mpxa implementation does not provide reconstruction of the flux on internal
        # faces, so we set the internal face values to zero also in the PorePy
        # implementation.
        internal_face = np.logical_not(grid_pp.tags["domain_boundary_faces"])
        if (
            attribute == "bound_pressure_cell"
            or attribute == "bound_pressure_face"
            or attribute == "bound_pressure_vector_source"
        ):
            m_pp[internal_face, :] = 0.0

        _compare_matrices(m_pp, m_cpp)
