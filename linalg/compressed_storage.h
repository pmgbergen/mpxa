/* Header file for the sparse matrix utilities.

*/
#ifndef LINALG_SPARSE_MATRIX_H
#define LINALG_SPARSE_MATRIX_H

class CompressedDataStorage {
    public:
        // Constructor for an empty matrix.
        CompressedDataStorage(int num_rows, int num_cols) ;

        // Constructor for a matrix with indices and values given.
        CompressedDataStorage(int num_rows, int num_cols, int* row_ptr, int* col_idx, double* values);

        // Destructor
        ~CompressedDataStorage();

        int num_rows();
        int num_cols();
    private: 
        int m_num_rows;
        int m_num_cols;
        int* m_row_ptr;
        int* m_col_idx;
        double* m_values;
};
#endif  // LINALG_SPARSE_MATRIX_H