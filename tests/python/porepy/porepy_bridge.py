import porepy as pp
import mpxa


def _sparse_matrix_conversion(sparse_matrix):
    """Convert a sparse matrix to the mpxa format."""
    return mpxa.CompressedDataStorageInt(
        sparse_matrix.shape[0],
        sparse_matrix.shape[1],
        sparse_matrix.indptr,
        sparse_matrix.indices,
        sparse_matrix.data,
    )


def grid_conversion(source_grid):
    dim = source_grid.dim
    nodes = source_grid.nodes.T
    cell_faces = _sparse_matrix_conversion(source_grid.cell_faces)
    face_nodes = _sparse_matrix_conversion(source_grid.face_nodes)
    target_grid = mpxa.Grid(dim, nodes, cell_faces, face_nodes)
    target_grid.set_cell_volumes(source_grid.cell_volumes)
    target_grid.set_face_areas(source_grid.face_areas)
    target_grid.set_face_normals(source_grid.face_normals.T)
    target_grid.set_cell_centers(source_grid.cell_centers.T)
    target_grid.set_face_centers(source_grid.face_centers.T)
    return target_grid


def convert_tensor(T: pp.SecondOrderTensor, dim: int):
    """Convert a SecondOrderTensor to the mpxa format."""
    return mpxa.SecondOrderTensor(
        dim,
        T.values[0, 0].size,
        T.values[0, 0],
        # T.values[1, 1],
        # T.values[2, 2],
        # T.values[0, 1],
        # T.values[0, 2],
        # T.values[1, 2],
    )


def convert_bc(bc: pp.BoundaryCondition):
    """Convert a BoundaryCondition to the mpxa format."""
    bc_map: dict[int, mpxa.BoundaryCondition] = {}

    for fi in range(bc.num_faces):
        if bc.is_dir[fi]:
            bc_map[fi] = mpxa.BoundaryCondition.Dirichlet
        elif bc.is_neu[fi]:
            bc_map[fi] = mpxa.BoundaryCondition.Neumann
        elif bc.is_rob[fi]:
            bc_map[fi] = mpxa.BoundaryCondition.Robin
        else:
            # This should be an internal face.
            pass

    return bc_map
