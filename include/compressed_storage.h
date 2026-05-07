/* Header file for the sparse matrix utilities.

*/
#ifndef LINALG_SPARSE_MATRIX_H
#define LINALG_SPARSE_MATRIX_H

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

template <typename T>
class CompressedDataStorage
{
   public:
    // Owning constructor: takes vectors by value (moved in) and owns the data.
    CompressedDataStorage(const int num_rows, const int num_cols, std::vector<int> row_ptr,
                          std::vector<int> col_idx, std::vector<T> values,
                          const bool construct_csc = false);

    // Non-owning constructor: borrows raw pointers without copying.
    // The caller must ensure the pointed-to arrays outlive this object.
    CompressedDataStorage(const int num_rows, const int num_cols,
                          const int* row_ptr, std::size_t row_ptr_size,
                          const int* col_idx, std::size_t col_idx_size,
                          const T* values, std::size_t values_size,
                          const bool construct_csc = false);

    // Factory: construct from non-owning CSC arrays.  The CSR format is built
    // internally (allocated once).  The caller must ensure the CSC arrays
    // outlive this object.
    static std::shared_ptr<CompressedDataStorage<T>> from_csc(
        int num_rows, int num_cols,
        const int* col_ptr, std::size_t col_ptr_size,
        const int* row_idx, std::size_t row_idx_size,
        const T* values, std::size_t values_size);

    ~CompressedDataStorage() = default;

    int num_rows() const;
    int num_cols() const;

    // Return non-owning views of the primary CSR arrays.
    std::span<const int> row_ptr() const;
    std::span<const int> col_idx() const;
    std::span<const T> data() const;

    // Getter for columns in a row, returns an immutable view of the data.
    std::span<const int> cols_in_row(int row) const;
    // Getter of rows in a column, creates a new vector since the underlying data is
    // not contiguous.
    std::vector<int> rows_in_col(int col) const;
    // Getter for values, returns a copy. Use data() to access the underlying span.
    std::vector<T> values() const;
    T value(const int row, const int col) const;

   private:
    void build_csc();

    int m_num_rows;
    int m_num_cols;

    // Owning storage — populated only by the owning constructor.
    std::vector<int> m_row_ptr_owned;
    std::vector<int> m_col_idx_owned;
    std::vector<T> m_values_owned;

    // Non-owning views into either m_*_owned or external caller-managed data.
    // Always valid after construction.
    std::span<const int> m_row_ptr;
    std::span<const int> m_col_idx;
    std::span<const T> m_values;

    std::vector<int> m_col_ptr;   // For CSC format (owned, built from CSR)
    std::vector<int> m_row_idx;   // For CSC format (owned, built from CSR)
    std::vector<T> m_values_csc;  // For CSC format (owned, built from CSR)

    // Non-owning views into CSC data (either m_col_ptr/m_row_idx/m_values_csc
    // or externally-managed arrays provided via from_csc()).
    std::span<const int> m_col_ptr_view;
    std::span<const int> m_row_idx_view;
    std::span<const T> m_values_csc_view;

    bool m_csc_constructed = false;
};

#endif  // LINALG_SPARSE_MATRIX_H