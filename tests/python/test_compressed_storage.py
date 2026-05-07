"""Test of the python bindings for compressed storage classes."""

import scipy.sparse as sps
import numpy as np
import pytest

from mpxa import _mpxa
from mpxa.porepy_bridge import convert_matrix_mpxa_to_scipy


@pytest.mark.parametrize("fmt", [int, float])
def test_storage(fmt):
    # Create a 4 x 3 sparse matrix with a few non-zero elements
    indptr = np.array([0, 2, 3, 3, 4], dtype=np.int32)
    indices = np.array([0, 2, 1, 0], dtype=np.int32)
    data_dtype = np.int32 if fmt is int else float
    data = np.array([1, 2, 3, 4], dtype=data_dtype)

    if fmt is int:
        storage_class = _mpxa.CompressedDataStorageInt
    else:
        storage_class = _mpxa.CompressedDataStorageDouble

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


@pytest.mark.parametrize("storage_class,data_dtype", [
    (_mpxa.CompressedDataStorageDouble, np.float64),
    (_mpxa.CompressedDataStorageInt, np.int32),
])
def test_accessors_return_numpy_arrays(storage_class, data_dtype):
    """row_ptr, col_idx, and data must return numpy arrays, not Python lists."""
    indptr = np.array([0, 2, 3, 3, 4], dtype=np.int32)
    indices = np.array([0, 2, 1, 0], dtype=np.int32)
    data = np.array([1, 2, 3, 4], dtype=data_dtype)
    cpp_mat = storage_class(4, 3, indptr, indices, data, False)

    assert isinstance(cpp_mat.row_ptr(), np.ndarray)
    assert isinstance(cpp_mat.col_idx(), np.ndarray)
    assert isinstance(cpp_mat.data(), np.ndarray)

    assert cpp_mat.row_ptr().dtype == np.int32
    assert cpp_mat.col_idx().dtype == np.int32
    assert cpp_mat.data().dtype == data_dtype


@pytest.mark.parametrize("storage_class,data_dtype", [
    (_mpxa.CompressedDataStorageDouble, np.float64),
    (_mpxa.CompressedDataStorageInt, np.int32),
])
def test_accessors_share_memory_with_cpp_object(storage_class, data_dtype):
    """Arrays returned by row_ptr/col_idx/data must share memory with the C++ object.
    The base of each array is the CompressedDataStorage Python wrapper (not a copy).
    """
    indptr = np.array([0, 2, 3, 3, 4], dtype=np.int32)
    indices = np.array([0, 2, 1, 0], dtype=np.int32)
    data = np.array([1, 2, 3, 4], dtype=data_dtype)
    cpp_mat = storage_class(4, 3, indptr, indices, data, False)

    rp = cpp_mat.row_ptr()
    ci = cpp_mat.col_idx()
    d = cpp_mat.data()

    # The returned arrays must be views (not owning copies).
    # numpy arrays that own their data have base=None; views have a non-None base.
    assert rp.base is not None, "row_ptr() should return a view, not a copy"
    assert ci.base is not None, "col_idx() should return a view, not a copy"
    assert d.base is not None, "data() should return a view, not a copy"

    # Values must match the input arrays
    np.testing.assert_array_equal(rp, indptr)
    np.testing.assert_array_equal(ci, indices)
    np.testing.assert_array_equal(d, data)


@pytest.mark.parametrize("storage_class,data_dtype,expected_dtype", [
    (_mpxa.CompressedDataStorageDouble, np.float64, np.float64),
    (_mpxa.CompressedDataStorageInt, np.int32, np.int32),
])
def test_convert_mpxa_to_scipy_correctness(storage_class, data_dtype, expected_dtype):
    """convert_matrix_mpxa_to_scipy must produce a matrix equivalent to the
    reference scipy CSR matrix, for both int and float storage types."""
    indptr = np.array([0, 2, 3, 3, 4], dtype=np.int32)
    indices = np.array([0, 2, 1, 0], dtype=np.int32)
    data = np.array([1, 2, 3, 4], dtype=data_dtype)
    num_rows, num_cols = 4, 3

    cpp_mat = storage_class(num_rows, num_cols, indptr, indices, data, False)
    result = convert_matrix_mpxa_to_scipy(cpp_mat)
    reference = sps.csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))

    assert result.shape == reference.shape
    assert result.dtype == expected_dtype
    np.testing.assert_array_equal(result.toarray(), reference.toarray())


def test_convert_mpxa_to_scipy_explicit_zeros():
    """Matrices with explicit zeros must be correctly converted: the result must
    match a reference matrix with those zeros removed."""
    # Build a matrix that has an explicit zero at (0, 0).
    indptr = np.array([0, 2, 3, 3, 4], dtype=np.int32)
    indices = np.array([0, 2, 1, 0], dtype=np.int32)
    data = np.array([0.0, 2.0, 3.0, 4.0], dtype=np.float64)  # explicit zero at (0,0)
    num_rows, num_cols = 4, 3

    cpp_mat = _mpxa.CompressedDataStorageDouble(num_rows, num_cols, indptr, indices, data, False)
    result = convert_matrix_mpxa_to_scipy(cpp_mat)

    # The explicit zero should have been removed by eliminate_zeros().
    assert result.nnz == 3
    expected = sps.csr_matrix(([2.0, 3.0, 4.0],
                                [2, 1, 0],
                                [0, 1, 2, 2, 3]),
                               shape=(num_rows, num_cols))
    np.testing.assert_array_equal(result.toarray(), expected.toarray())


def test_convert_mpxa_to_scipy_no_spurious_copies():
    """When the matrix has no explicit zeros, the CSR arrays returned by scipy
    should share memory with the C++ binding arrays (copy=False path)."""
    indptr = np.array([0, 2, 3, 3, 4], dtype=np.int32)
    indices = np.array([0, 2, 1, 0], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    num_rows, num_cols = 4, 3

    cpp_mat = _mpxa.CompressedDataStorageDouble(num_rows, num_cols, indptr, indices, data, False)

    # Capture the binding arrays before conversion
    rp_binding = cpp_mat.row_ptr()
    ci_binding = cpp_mat.col_idx()
    d_binding = cpp_mat.data()

    result = convert_matrix_mpxa_to_scipy(cpp_mat)

    # After conversion (no explicit zeros, so eliminate_zeros is a no-op),
    # the scipy matrix arrays should share memory with the binding arrays.
    assert np.shares_memory(result.indptr, rp_binding), \
        "result.indptr should share memory with row_ptr() binding array"
    assert np.shares_memory(result.indices, ci_binding), \
        "result.indices should share memory with col_idx() binding array"
    assert np.shares_memory(result.data, d_binding), \
        "result.data should share memory with data() binding array"
