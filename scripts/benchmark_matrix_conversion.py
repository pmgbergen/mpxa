"""Benchmark: convert_matrix_mpxa_to_scipy for matrices of increasing size.

Measures the round-trip time of converting a CompressedDataStorage object
(built via from_csc) to a SciPy CSR matrix via convert_matrix_mpxa_to_scipy.
Run with:
    python scripts/benchmark_matrix_conversion.py
"""

import time

import numpy as np
import scipy.sparse as sps

from mpxa import _mpxa
from mpxa.porepy_bridge import convert_matrix_mpxa_to_scipy


def make_random_csc(num_rows: int, num_cols: int, nnz: int, rng: np.random.Generator):
    """Return (col_ptr, row_idx, values) for a random CSC matrix."""
    cols = rng.integers(0, num_cols, size=nnz)
    rows = rng.integers(0, num_rows, size=nnz)
    vals = rng.standard_normal(nnz)

    mat = sps.coo_matrix((vals, (rows, cols)), shape=(num_rows, num_cols)).tocsc()
    mat.sum_duplicates()

    col_ptr = mat.indptr.astype(np.int32)
    row_idx = mat.indices.astype(np.int32)
    values = mat.data.astype(np.float64)
    return col_ptr, row_idx, values, mat.shape


def benchmark_size(num_rows: int, num_cols: int, nnz: int, repeats: int = 20) -> float:
    """Return mean conversion time in seconds over *repeats* runs."""
    rng = np.random.default_rng(0)
    col_ptr, row_idx, values, shape = make_random_csc(num_rows, num_cols, nnz, rng)

    cpp_mat = _mpxa.CompressedDataStorageDouble.from_csc(
        shape[0], shape[1], col_ptr, row_idx, values
    )

    # Warm-up
    convert_matrix_mpxa_to_scipy(cpp_mat)

    start = time.perf_counter()
    for _ in range(repeats):
        convert_matrix_mpxa_to_scipy(cpp_mat)
    elapsed = time.perf_counter() - start
    return elapsed / repeats


def main():
    print(f"{'rows':>8} {'cols':>8} {'nnz':>10}  {'mean µs':>10}")
    print("-" * 44)

    sizes = [
        (100,   100,    500),
        (1_000, 1_000,  5_000),
        (5_000, 5_000,  50_000),
        (20_000, 20_000, 200_000),
    ]

    for num_rows, num_cols, nnz in sizes:
        mean_s = benchmark_size(num_rows, num_cols, nnz)
        print(f"{num_rows:>8} {num_cols:>8} {nnz:>10}  {mean_s * 1e6:>10.1f}")


if __name__ == "__main__":
    main()
