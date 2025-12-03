import porepy as pp
import numpy as np
import scipy.sparse as sps



def test_sparse_matrix_conversion():
    np.random.seed(0)

    # Create a sparse matrix
    mat_sps = sps.random(10, 12, density=0.3, format="csr")

    mat_mpxa = mpxa.convert_matrix(mat_sps)

    # Check the shape
    assert mat_mpxa.num_rows() == mat_sps.shape[0]
    assert mat_mpxa.num_cols() == mat_sps.shape[1]

    # Check the data by looping over all entries. This is not efficient, but will check
    # both zeros and non-zeros.
    for i in range(mat_sps.shape[0]):
        for j in range(mat_sps.shape[1]):
            assert mat_mpxa.value(i, j) == mat_sps[i, j]
