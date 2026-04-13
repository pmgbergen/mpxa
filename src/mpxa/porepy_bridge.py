import porepy as pp
from mpxa import _mpxa
import numpy as np
import scipy.sparse as sps


def convert_matrix_scipy_to_mpxa(
    sparse_matrix: sps.csr_matrix | sps.csr_array, csc: bool = False
) -> _mpxa.CompressedDataStorageDouble | _mpxa.CompressedDataStorageInt:
    """
    Parameters:
        sparse_matrix: Input matrix in the csr format.
        csc: Flag to indicate that the underlying c++ storage should store both the csc
            and csr formats. The input matrix must still be passed in the csr format.

    """
    assert sparse_matrix.format == "csr"
    if sparse_matrix.data.dtype == int:
        matrix_class = _mpxa.CompressedDataStorageInt
    elif sparse_matrix.data.dtype == float:
        matrix_class = _mpxa.CompressedDataStorageDouble
    else:
        raise ValueError(
            f"Unsupported data type {sparse_matrix.data.dtype} for sparse matrix."
        )

    return matrix_class(
        sparse_matrix.shape[0],
        sparse_matrix.shape[1],
        np.ascontiguousarray(sparse_matrix.indptr),
        np.ascontiguousarray(sparse_matrix.indices),
        np.ascontiguousarray(sparse_matrix.data),
        csc,
    )


def convert_matrix_mpxa_to_scipy(
    mpxa_matrix: _mpxa.CompressedDataStorageInt | _mpxa.CompressedDataStorageDouble,
) -> sps.csr_matrix:
    """Convert an mpxa.CompressedDataStorage to a scipy sparse matrix."""
    if isinstance(mpxa_matrix, _mpxa.CompressedDataStorageInt):
        dtype = int
    elif isinstance(mpxa_matrix, _mpxa.CompressedDataStorageDouble):
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


def convert_vector_source_mpxa_to_scipy(
    mpxa_matrix: _mpxa.CompressedDataStorageDouble, ambient_dim: int
) -> sps.csr_matrix:
    """Special treatment of the `vector_source` matrix. In mpxa (C++ code), the ambient
    dimension for the vector source discretization is always set to 3. It breaks with
    the standard PorePy implementation in the case where the computational domain is 1D
    or 2D.

    This function trims the extra dimension to stay consistent with PorePy.
    """
    result_mat = convert_matrix_mpxa_to_scipy(mpxa_matrix=mpxa_matrix)
    if ambient_dim == 3:
        return result_mat
    mask = np.ones(result_mat.shape[1], dtype=bool)
    if ambient_dim <= 2:
        # Skipping each 3rd element (z-axis).
        mask[2::3] = False
    if ambient_dim <= 1:
        # Skipping each 2nd element (y-axis).
        mask[1::3] = False

    return result_mat[:, mask]


def convert_grid_to_mpxa(source_grid: pp.Grid) -> _mpxa.Grid:
    dim = source_grid.dim
    nodes = source_grid.nodes.T
    cell_faces = convert_matrix_scipy_to_mpxa(source_grid.cell_faces.tocsr(), csc=True)
    face_nodes = convert_matrix_scipy_to_mpxa(source_grid.face_nodes.tocsr(), csc=True)
    target_grid = _mpxa.Grid(dim, nodes, cell_faces, face_nodes)
    target_grid.set_cell_volumes(np.ascontiguousarray(source_grid.cell_volumes))
    target_grid.set_face_areas(np.ascontiguousarray(source_grid.face_areas))
    target_grid.set_face_normals(np.ascontiguousarray(source_grid.face_normals.T))
    target_grid.set_cell_centers(np.ascontiguousarray(source_grid.cell_centers.T))
    target_grid.set_face_centers(np.ascontiguousarray(source_grid.face_centers.T))
    return target_grid


def convert_tensor_to_mpxa(
    T: pp.SecondOrderTensor, dim: int
) -> _mpxa.SecondOrderTensor:
    """Convert a SecondOrderTensor to the mpxa format."""

    # The underlying array should be contiguous.
    values = np.ascontiguousarray(T.values)

    if dim == 0:
        return _mpxa.SecondOrderTensor(0, values[0, 0].size, values[0, 0])
    elif dim == 1:
        return _mpxa.SecondOrderTensor(1, values[0, 0].size, values[0, 0])
    elif dim == 2:
        if not np.allclose(values[0, 1], 0):
            # This is a full tensor in 2d.
            return _mpxa.SecondOrderTensor(
                dim,
                values[0, 0].size,
                values[0, 0],
                values[1, 1],
                values[0, 1],
            )
        elif not np.allclose(values[0, 0], values[1, 1]):
            # This is an anisotropic, but diagonal tensor in 2d.
            return _mpxa.SecondOrderTensor(
                dim,
                values[0, 0].size,
                values[0, 0],
                values[1, 1],
            )
        else:
            # This is an isotropic tensor in 2d.
            return _mpxa.SecondOrderTensor(
                dim,
                values[0, 0].size,
                values[0, 0],
            )
    elif dim == 3:
        if not (
            np.allclose(values[0, 1], 0)
            | np.allclose(values[0, 2], 0)
            | np.allclose(values[1, 2], 0)
        ):
            # This is a full tensor in 3d.
            return _mpxa.SecondOrderTensor(
                dim,
                values[0, 0].size,
                values[0, 0],
                values[1, 1],
                values[2, 2],
                values[0, 1],
                values[0, 2],
                values[1, 2],
            )
        elif not (
            np.allclose(values[0, 0], values[1, 1])
            | np.allclose(values[0, 0], values[2, 2])
        ):
            # This is an anisotropic, but diagonal tensor in 3d.
            return _mpxa.SecondOrderTensor(
                dim,
                values[0, 0].size,
                values[0, 0],
                values[1, 1],
                values[2, 2],
            )
        else:
            # This is an isotropic tensor in 3d.
            return _mpxa.SecondOrderTensor(
                dim,
                values[0, 0].size,
                values[0, 0],
            )
    else:
        raise ValueError(
            f"Unsupported dimension {dim} for SecondOrderTensor conversion."
        )


def convert_bc_to_mpxa(bc: pp.BoundaryCondition) -> dict[int, _mpxa.BoundaryCondition]:
    """Convert a BoundaryCondition to the mpxa format."""
    bc_map: dict[int, _mpxa.BoundaryCondition] = {}

    for fi in range(bc.num_faces):
        if bc.is_dir[fi]:
            bc_map[fi] = _mpxa.BoundaryCondition.Dirichlet
        elif bc.is_neu[fi]:
            bc_map[fi] = _mpxa.BoundaryCondition.Neumann
        elif bc.is_rob[fi]:
            bc_map[fi] = _mpxa.BoundaryCondition.Robin
        else:
            # This should be an internal face.
            pass

    return bc_map
