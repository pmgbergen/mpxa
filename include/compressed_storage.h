/* Header file for the sparse matrix utilities.

*/
#ifndef LINALG_SPARSE_MATRIX_H
#define LINALG_SPARSE_MATRIX_H

#include <algorithm>
#include <vector>

template <typename T>
class CompressedDataStorage
{
   public:
    // Constructor for a matrix with indices and values given.
    CompressedDataStorage(const int num_rows, const int num_cols, const std::vector<int>& row_ptr,
                          const std::vector<int>& col_idx, const std::vector<T>& values);

    // Destructor
    ~CompressedDataStorage();

    const int num_rows();
    const int num_cols();

    // Getters for the compressed data storage. These return *copies* of the data. TODO!
    std::vector<int> cols_in_row(int row);
    std::vector<int> rows_in_col(int col);  // Change from int* to std::vector<int>

    std::vector<T> values();
    T value(const int row, const int col);

   private:
    int m_num_rows;
    int m_num_cols;
    std::vector<int> m_row_ptr;
    std::vector<int> m_col_idx;
    std::vector<T> m_values;  // Change from T* to std::vector<T>
};

#endif  // LINALG_SPARSE_MATRIX_H