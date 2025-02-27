/* Header file for the sparse matrix utilities.

*/
#ifndef LINALG_SPARSE_MATRIX_H
#define LINALG_SPARSE_MATRIX_H

template <typename T>
class CompressedDataStorage
{
   public:
    // Constructor for an empty matrix.
    CompressedDataStorage(int num_rows, int num_cols);

    // Constructor for a matrix with indices and values given.
    CompressedDataStorage(int num_rows, int num_cols, int* row_ptr, int* col_idx, T* values);

    // Destructor
    ~CompressedDataStorage();

    const int num_rows();
    const int num_cols();

    // Getters for the compressed data storage. These return *copies* of the data. TODO!
    int* cols_in_row(int row);
    int* rows_in_col(int col);

    const T* values();
    T* values_in_row(int row);

   private:
    int m_num_rows;
    int m_num_cols;
    int* m_row_ptr;
    int* m_col_idx;
    double* m_values;
};
#endif  // LINALG_SPARSE_MATRIX_H