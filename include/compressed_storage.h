/* Header file for the sparse matrix utilities.

*/
#ifndef LINALG_SPARSE_MATRIX_H
#define LINALG_SPARSE_MATRIX_H

#include <algorithm>
#include <vector>
#include <span>

template <typename T>
class CompressedDataStorage
{
   public:
    // Constructor for a matrix with indices and values given.
    CompressedDataStorage(const int num_rows, const int num_cols, std::vector<int> row_ptr,
                          std::vector<int> col_idx, std::vector<T> values,
                          const bool construct_csc = false);

    // Destructor
    ~CompressedDataStorage();

    const int num_rows();
    const int num_cols();

    const std::vector<int>& row_ptr() const;
    const std::vector<int>& col_idx() const;
    const std::vector<T>& data() const;

    // Getters for the compressed data storage.
    
    // Getter for columns in a row, returns an immutable view of the data.
    std::span<const int> cols_in_row(int row);
    // Getter of rows in a column, creates a new vector, since the underlying data is
    // not contiguious.
    std::vector<int> rows_in_col(int col);
    // Getter for values, returns a copy of the values. Use `data` to access the
    // underlying data.
    std::vector<T> values();
    T value(const int row, const int col);

   private:
    int m_num_rows;
    int m_num_cols;
    std::vector<int> m_row_ptr;
    std::vector<int> m_col_idx;
    std::vector<T> m_values;      // Change from T* to std::vector<T>
    std::vector<int> m_col_ptr;   // For CSC format
    std::vector<int> m_row_idx;   // For CSC format
    std::vector<T> m_values_csc;  // For CSC format
    // This class stores a sparse matrix in the CSR format. Optionally, the CSC indices
    // can be constructed alongside. 
    bool m_csc_constructed = false;
};

#endif  // LINALG_SPARSE_MATRIX_H