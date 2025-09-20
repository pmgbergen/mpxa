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

    assert m_0.shape == m_1.shape, f"Shape mismatch: {m_0.shape} vs {m_1.shape}"

    diff = m_0 - m_1
    assert np.allclose(diff.data, 0, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("g_pp", grid_list)
@pytest.mark.parametrize(
    "tensor_func",
    [
        isotropic_tensor,
        anisotropic_tensor,
        full_tensor,
    ],
)
@pytest.mark.parametrize("discr_type", ["tpfa", "mpfa"])
def test_tpfa(g_pp, tensor_func, discr_type):
    K_pp = tensor_func(g_pp)
    bc_pp = pp.BoundaryCondition(g_pp)
    bc_pp.is_dir[0] = True
    bc_pp.is_neu[0] = False

    K = porepy_bridge.convert_tensor(K_pp, g_pp.dim)
    bc = porepy_bridge.convert_bc(bc_pp)
    g = porepy_bridge.convert_grid(g_pp)

    key = "flow"

    if discr_type == "tpfa":
        discr_pp = pp.Tpfa(key)
    elif discr_type == "mpfa":
        discr_pp = pp.Mpfa(key)

    # Set the parameters for the discretization. Note that the ambient dimension for the
    # vector source discretization is always set to 3. This follows the C++ code, but
    # breaks with the standard PorePy implementation in the case where the computational
    # domain is 2D. Todo, I guess.
    data = {
        pp.PARAMETERS: {
            key: {"second_order_tensor": K_pp, "bc": bc_pp, "ambient_dimension": 3}
        },
        pp.DISCRETIZATION_MATRICES: {key: {}},
    }
    discr_pp.discretize(g_pp, data)

    if discr_type == "tpfa":
        discr_cpp = mpxa.tpfa(g, K, bc)
    elif discr_type == "mpfa":
        discr_cpp = mpxa.mpfa(g, K, bc)

    for attribute in [
        "vector_source",
        "flux",
        "bound_flux",
        "bound_pressure_face",
        "bound_pressure_cell",
        "bound_pressure_vector_source",
    ]:
        m_0 = getattr(discr_cpp, attribute)
        m_1 = data[pp.DISCRETIZATION_MATRICES][key][attribute]
        m_0 = porepy_bridge.convert_matrix(m_0)
        print(m_1.shape)

        internal_face = np.logical_not(g_pp.tags["domain_boundary_faces"])
        # The mpxa implementation does not provide reconstruction of the flux on internal
        # faces, so we set the internal face values to zero also in the PorePy
        # implementation.
        if (
            attribute == "bound_pressure_cell"
            or attribute == "bound_pressure_face"
            or attribute == "bound_pressure_vector_source"
        ):
            m_1[internal_face, :] = 0.0

        _compare_matrices(m_0, m_1)


# test_tpfa(grid_list[0], isotropic_tensor)
# print("Isotropic tensor test passed.")
# test_tpfa(grid_list[0], anisotropic_tensor)
