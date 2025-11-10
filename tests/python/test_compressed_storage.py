"""Test of the python bindings for compressed storage classes."""

import scipy.sparse as sps
import numpy as np
import pytest

import mpxa


@pytest.mark.parametrize("fmt", [int, float])
def test_storage(fmt):
    # Create a 4 x 3 sparse matrix with a few non-zero elements
    indptr = np.array([0, 2, 3, 3, 4], dtype=int)
    indices = np.array([0, 2, 1, 0], dtype=int)
    data = np.array([1, 2, 3, 4], dtype=fmt)

    if fmt is int:
        storage_class = mpxa.CompressedDataStorageInt
    else:
        storage_class = mpxa.CompressedDataStorageDouble

    num_rows = 4
    num_cols = 3
    # Do not test the case with CSC format for now (hence the last argument is False).
    cpp_mat = storage_class(num_rows, num_cols, indptr, indices, data, False)
    sps_mat = sps.csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))

    row, col, val = sps.find(sps_mat)

    for r, c, v in zip(row, col, val):
        assert cpp_mat.value(r, c) == v

    # Check the number of rows and columns
    assert cpp_mat.num_rows() == num_rows
    assert cpp_mat.num_cols() == num_cols
    # Check the data array
    np.testing.assert_allclose(cpp_mat.values(), data)
