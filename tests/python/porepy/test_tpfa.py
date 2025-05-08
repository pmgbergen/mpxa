import porepy as pp
import numpy as np
import pytest
import porepy_bridge
import mpxa

import scipy.sparse as sps


grid_list = [
    pp.CartGrid(np.array([3, 3])),
    pp.CartGrid(np.array([3, 3, 3])),
    pp.StructuredTriangleGrid(np.array([3, 3])),
    pp.StructuredTetrahedralGrid(np.array([3, 3, 3])),
]
for g in grid_list:
    g.compute_geometry()


def isotropic_tensor(g):
    num_cells = g.num_cells
    e = np.ones(num_cells)
    return pp.SecondOrderTensor(e)


def anisotropic_tensor(g):
    num_cells = g.num_cells
    e = np.ones(num_cells)
    return pp.SecondOrderTensor(e, 2 * e, 3 * e)


def full_tensor(g):
    num_cells = g.num_cells
    e = np.ones(num_cells)
    return pp.SecondOrderTensor(e, 2 * e, 3 * e, 0.1 * e, 0.2 * e, 0.3 * e)


def _compare_matrices(m_0: mpxa.CompressedDataStorageDouble, m_1: sps.spmatrix):
    """Compare two matrices for equality."""
    assert m_0.num_rows() == m_1.shape[0]
    assert m_0.num_cols() == m_1.shape[1]
    assert np.array_equal(m_0.row_ptr(), m_1.indptr)
    assert np.array_equal(m_0.col_idx(), m_1.indices)
    assert np.allclose(m_0.data(), m_1.data)


@pytest.mark.parametrize("g_pp", grid_list)
@pytest.mark.parametrize(
    "tensor_func",
    [
        isotropic_tensor,
        anisotropic_tensor,
        full_tensor,
    ],
)
def test_tpfa(g_pp, tensor_func):
    K_pp = tensor_func(g_pp)
    bc_pp = pp.BoundaryCondition(g_pp)
    bc_pp.is_dir[0] = True
    bc_pp.is_neu[0] = False

    K = porepy_bridge.convert_tensor(K_pp, g_pp.dim)
    bc = porepy_bridge.convert_bc(bc_pp)
    g = porepy_bridge.convert_grid(g_pp)

    key = "flow"

    discr_pp = pp.Tpfa(key)

    data = {
        pp.PARAMETERS: {key: {"second_order_tensor": K_pp, "bc": bc_pp}},
        pp.DISCRETIZATION_MATRICES: {key: {}},
    }
    discr_pp.discretize(g_pp, data)

    flux = data[pp.DISCRETIZATION_MATRICES][key]["flux"]

    discr_cpp = mpxa.tpfa(g, K, bc)

    for attribute in ["flux"]:
        m_0 = getattr(discr_cpp, attribute)
        m_1 = data[pp.DISCRETIZATION_MATRICES][key][attribute]
        _compare_matrices(m_0, m_1)
