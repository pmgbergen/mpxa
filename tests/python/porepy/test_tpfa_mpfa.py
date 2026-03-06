"""Tests for the finite volume discretizations in PorePy against the mpxa bindings.

The test assumes that PorePy is installed.

"""

from typing import Callable

import porepy as pp
import numpy as np
import pytest
import mpxa

import scipy.sparse as sps

# The lines below generate a 2D grid, tilted in a 3D domain. MPFA and TPFA tests fail
# with this grid. It probably should be stored differently, in a less ugly way (TODO).
nodes = np.array(
    [
        [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        [
            1.0,
            0.5,
            0.5,
            0.0,
            1.0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.0,
            0.0,
            1.0,
            0.5,
            0.5,
            0.0,
        ],
    ]
)
face_nodes_data = np.array(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
)
faces_nodes_col_idx = np.array(
    [
        0,
        4,
        2,
        7,
        3,
        10,
        5,
        12,
        9,
        14,
        11,
        15,
        1,
        0,
        3,
        2,
        8,
        5,
        11,
        9,
        13,
        12,
        15,
        14,
        6,
        4,
        1,
        6,
        10,
        7,
        8,
        13,
    ]
)
faces_nodes_row_ptr = np.array(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
)
cell_faces_data = np.array([-1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
cell_faces_col_idx = np.array([0, 6, 12, 13, 1, 2, 7, 14, 3, 8, 10, 15, 4, 5, 9, 11])
cell_faces_row_ptr = np.array([0, 4, 8, 12, 16])


@pytest.fixture(
    scope="module",
    params=[
        pp.CartGrid(np.array([3, 3])),  # 2D cartesian grid.
        pp.CartGrid(np.array([3, 3, 3])),  # 3D cartesian grid.
        pp.StructuredTriangleGrid(np.array([3, 3])),  # 2D simplex grid.
        pp.StructuredTetrahedralGrid(np.array([3, 3, 3])),  # 3D simplex grid.
        pp.TensorGrid(x=np.linspace(0, 3, num=10, endpoint=True)),  # 1D line grid.
        pp.PointGrid(pt=np.array([0.5, 0.5, 0.5])),  # 0D point grid.
        #
        pp.Grid(
            dim=2,
            nodes=nodes,
            face_nodes=sps.csc_matrix(
                (face_nodes_data, faces_nodes_col_idx, faces_nodes_row_ptr)
            ),
            cell_faces=sps.csc_matrix(
                (cell_faces_data, cell_faces_col_idx, cell_faces_row_ptr)
            ),
            name="2d in 3d",
        ),
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
        "vector_source",
        "flux",
        "bound_flux",
        "bound_pressure_face",
        "bound_pressure_cell",
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
