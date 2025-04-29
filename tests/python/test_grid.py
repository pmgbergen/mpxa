import pytest
import numpy as np
import scipy.sparse as sps

import mpxa


def test_grid_bindings():
    fn_nodes = np.array(
        [0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 8, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8]
    )
    fn_faces = np.repeat(np.arange(12), 2)
    face_nodes = sps.coo_matrix(
        (np.ones(24), (fn_nodes, fn_faces)), shape=(9, 12)
    ).tocsr()

    fn_cpp = mpxa.CompressedDataStorageInt(
        9, 12, face_nodes.indptr, face_nodes.indices, face_nodes.data
    )

    cf_faces = np.array([0, 1, 6, 8, 1, 2, 7, 9, 3, 4, 8, 10, 4, 5, 9, 11])
    cf_cells = np.repeat(np.arange(4), 4)
    cf_sgn = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
    cf_cpp = mpxa.CompressedDataStorageInt(12, 4, cf_faces, cf_cells, cf_sgn)

    nodes = np.array(
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]]
    ).astype(float)

    grid = mpxa.Grid(2, nodes, fn_cpp, cf_cpp)

    cell_volumes = np.ones(4, dtype=float)
    face_areas = np.ones(12, dtype=float)
    face_normals = np.array(
        [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    ).T

    cell_centers = np.array(
        [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]], dtype=float
    )
    face_centers = np.array(
        [
            [0, 0.5],
            [1, 0.5],
            [2, 0.5],
            [0, 1.5],
            [1, 1.5],
            [2, 1.5],
            [0.5, 0],
            [0.5, 1],
            [0.5, 1],
            [1.5, 1],
            [0.5, 2],
            [1.5, 2],
        ],
        dtype=float,
    )

    grid.set_cell_volumes(cell_volumes)
    grid.set_face_areas(face_areas)
    grid.set_face_normals(face_normals)
    grid.set_cell_centers(cell_centers)
    grid.set_face_centers(face_centers)

    np.testing.assert_allclose(grid.cell_volumes(), cell_volumes, rtol=1e-10)
    np.testing.assert_allclose(grid.face_areas(), face_areas, rtol=1e-10)
    np.testing.assert_allclose(grid.face_normals(), face_normals, rtol=1e-10)
    np.testing.assert_allclose(grid.cell_centers(), cell_centers, rtol=1e-10)
    np.testing.assert_allclose(grid.face_centers(), face_centers, rtol=1e-10)
    np.testing.assert_allclose(grid.nodes(), nodes, rtol=1e-10)
