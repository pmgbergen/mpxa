import numpy as np
import porepy as pp
import pytest

import mpxa


@pytest.fixture(scope="module")
def grids_2d():
    """Fixture to construct the grids and their porepy counterparts."""
    g_pp = pp.CartGrid(np.array([3, 3]))
    g_pp.compute_geometry()
    g = mpxa.convert_grid(g_pp)

    return (g, g_pp)


@pytest.fixture(scope="module")
def grids_3d():
    g_3d_pp = pp.CartGrid(np.array([3, 3, 3]))
    g_3d_pp.compute_geometry()
    g_3d = mpxa.convert_grid(g_3d_pp)
    return (g_3d, g_3d_pp)


@pytest.mark.parametrize("grid_func", ["grids_2d", "grids_3d"])
def test_cell_volumes(grid_func, request):
    """Test the cell volumes of the grid."""
    g, g_pp = request.getfixturevalue(grid_func)
    for i in range(g_pp.num_cells):
        assert np.allclose(g.cell_volume(i), g_pp.cell_volumes[i])


@pytest.mark.parametrize("grid_func", ["grids_2d", "grids_3d"])
def test_face_areas(grid_func, request):
    """Test the face areas of the grid."""
    g, g_pp = request.getfixturevalue(grid_func)
    for i in range(g_pp.num_faces):
        assert np.allclose(g.face_area(i), g_pp.face_areas[i])


@pytest.mark.parametrize("grid_func", ["grids_2d", "grids_3d"])
def test_face_normals(grid_func, request):
    """Test the face normals of the grid."""
    g, g_pp = request.getfixturevalue(grid_func)
    for i in range(g_pp.num_faces):
        assert np.allclose(g.face_normal(i), g_pp.face_normals[:, i])


@pytest.mark.parametrize("grid_func", ["grids_2d", "grids_3d"])
def test_nodes(grid_func, request):
    """Test the nodes of the grid."""
    g, g_pp = request.getfixturevalue(grid_func)
    assert np.allclose(g.nodes(), g_pp.nodes.T)


@pytest.mark.parametrize("grid_func", ["grids_2d", "grids_3d"])
def test_cell_centers(grid_func, request):
    """Test the cell centers of the grid."""
    g, g_pp = request.getfixturevalue(grid_func)
    for i in range(g_pp.num_cells):
        assert np.allclose(g.cell_center(i), g_pp.cell_centers[:, i])


# Topology
def _faces_of_node(g_pp, node):
    """Get the faces of a node."""
    return np.where(np.squeeze(g_pp.face_nodes[node].toarray()))[0]


def _nodes_of_face(g_pp, face):
    """Get the nodes of a face."""
    return np.where(np.squeeze(g_pp.face_nodes[:, face].toarray()))[0]


def _faces_of_cell(g_pp, cell):
    """Get the faces of a cell."""
    return np.where(np.squeeze(g_pp.cell_faces[:, cell].toarray()))[0]


def _cells_of_face(g_pp, face):
    """Get the cells of a face."""
    return np.where(np.squeeze(g_pp.cell_faces[face].toarray()))[0]


@pytest.mark.parametrize("grid_func", ["grids_2d", "grids_3d"])
def test_faces_of_node(grid_func, request):
    """Test the faces of a node."""
    g, g_pp = request.getfixturevalue(grid_func)
    for ni in range(g_pp.num_nodes):
        assert np.allclose(g.faces_of_node(ni), _faces_of_node(g_pp, ni), atol=1e-6), (
            f"Node {ni} faces do not match."
        )
    for fi in range(g_pp.num_faces):
        assert np.allclose(g.nodes_of_face(fi), _nodes_of_face(g_pp, fi), atol=1e-6), (
            f"Face {fi} nodes do not match."
        )


@pytest.mark.parametrize("grid_func", ["grids_2d", "grids_3d"])
def test_faces_of_cell(grid_func, request):
    """Test the faces of a cell."""
    g, g_pp = request.getfixturevalue(grid_func)
    for ci in range(g_pp.num_cells):
        assert np.allclose(g.faces_of_cell(ci), _faces_of_cell(g_pp, ci), atol=1e-6), (
            f"Cell {ci} faces do not match."
        )
    for fi in range(g_pp.num_faces):
        assert np.allclose(g.cells_of_face(fi), _cells_of_face(g_pp, fi), atol=1e-6), (
            f"Face {fi} cells do not match."
        )
