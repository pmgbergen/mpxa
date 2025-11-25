import porepy as pp
import mpxa
import numpy as np
import scipy.sparse as sps


def convert_matrix(sparse_matrix, csc: bool = False):
    """Convert a sparse matrix to the mpxa format."""

    if isinstance(sparse_matrix, (sps.spmatrix, sps.sparray)):
        return _convert_matrix_to_mpxa(sparse_matrix, csc)
    elif isinstance(
        sparse_matrix, (mpxa.CompressedDataStorageInt, mpxa.CompressedDataStorageDouble)
    ):
        return _convert_matrix_to_scipy(sparse_matrix)
    else:
        raise ValueError(
            f"Unsupported sparse matrix type {type(sparse_matrix)} for conversion."
        )


def _convert_matrix_to_mpxa(sparse_matrix, csc: bool = False):
    if sparse_matrix.data.dtype == int:
        matrix_class = mpxa.CompressedDataStorageInt
    elif sparse_matrix.data.dtype == float:
        matrix_class = mpxa.CompressedDataStorageDouble
    else:
        raise ValueError(
            f"Unsupported data type {sparse_matrix.data.dtype} for sparse matrix."
        )

    return matrix_class(
        sparse_matrix.shape[0],
        sparse_matrix.shape[1],
        sparse_matrix.indptr,
        sparse_matrix.indices,
        np.ascontiguousarray(sparse_matrix.data),
        csc,
    )


def _convert_matrix_to_scipy(mpxa_matrix):
    """Convert an mpxa.CompressedDataStorage to a scipy sparse matrix."""
    if isinstance(mpxa_matrix, mpxa.CompressedDataStorageInt):
        dtype = int
    elif isinstance(mpxa_matrix, mpxa.CompressedDataStorageDouble):
        dtype = float
    else:
        raise ValueError(
            f"Unsupported mpxa matrix type {type(mpxa_matrix)} for conversion."
        )

    return sps.csr_matrix(
        (mpxa_matrix.data(), mpxa_matrix.col_idx(), mpxa_matrix.row_ptr()),
        shape=(mpxa_matrix.num_rows(), mpxa_matrix.num_cols()),
        dtype=dtype,
    )


def convert_grid(source_grid):
    dim = source_grid.dim
    nodes = source_grid.nodes.T
    cell_faces = convert_matrix(source_grid.cell_faces.tocsr(), True)
    face_nodes = convert_matrix(source_grid.face_nodes.tocsr(), True)
    target_grid = mpxa.Grid(dim, nodes, cell_faces, face_nodes)
    target_grid.set_cell_volumes(source_grid.cell_volumes)
    target_grid.set_face_areas(source_grid.face_areas)
    target_grid.set_face_normals(source_grid.face_normals.T)
    target_grid.set_cell_centers(source_grid.cell_centers.T)
    target_grid.set_face_centers(source_grid.face_centers.T)
    return target_grid


def convert_tensor(T: pp.SecondOrderTensor, dim: int):
    """Convert a SecondOrderTensor to the mpxa format."""

    if dim == 2:
        if not np.allclose(T.values[0, 1], 0):
            # This is a full tensor in 2d.
            return mpxa.SecondOrderTensor(
                dim,
                T.values[0, 0].size,
                T.values[0, 0],
                T.values[1, 1],
                T.values[0, 1],
            )
        elif not np.allclose(T.values[0, 0], T.values[1, 1]):
            # This is an anisotropic, but diagonal tensor in 2d.
            return mpxa.SecondOrderTensor(
                dim,
                T.values[0, 0].size,
                T.values[0, 0],
                T.values[1, 1],
            )
        else:
            # This is an isotropic tensor in 2d.
            return mpxa.SecondOrderTensor(
                dim,
                T.values[0, 0].size,
                T.values[0, 0],
            )
    elif dim == 3:
        if not (
            np.allclose(T.values[0, 1], 0)
            | np.allclose(T.values[0, 2], 0)
            | np.allclose(T.values[1, 2], 0)
        ):
            # This is a full tensor in 3d.
            return mpxa.SecondOrderTensor(
                dim,
                T.values[0, 0].size,
                T.values[0, 0],
                T.values[1, 1],
                T.values[2, 2],
                T.values[0, 1],
                T.values[0, 2],
                T.values[1, 2],
            )
        elif not (
            np.allclose(T.values[0, 0], T.values[1, 1])
            | np.allclose(T.values[0, 0], T.values[2, 2])
        ):
            # This is an anisotropic, but diagonal tensor in 3d.
            return mpxa.SecondOrderTensor(
                dim,
                T.values[0, 0].size,
                T.values[0, 0],
                T.values[1, 1],
                T.values[2, 2],
            )
        else:
            # This is an isotropic tensor in 3d.
            return mpxa.SecondOrderTensor(
                dim,
                T.values[0, 0].size,
                T.values[0, 0],
            )
    else:
        raise ValueError(
            f"Unsupported dimension {dim} for SecondOrderTensor conversion."
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
