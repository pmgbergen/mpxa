#include <sparse_matrix.h>

class CompressedDataStorage {
    public:
        // Constructor for an empty matrix.
        CompressedDataStorage(int num_rows, int num_cols)
        : m_num_rows(num_rows)
        , m_num_cols(num_cols)
        , m_row_ptr(new int[num_rows + 1])
        , m_col_idx(new int[0])
        , m_values(new double[0]) {
            m_row_ptr[0] = 0;
        }

        // Constructor for a matrix with indices and values given.
        CompressedDataStorage(int num_rows, int num_cols, int* row_ptr, int* col_idx, double* values)
        : m_num_rows(num_rows)
        , m_num_cols(num_cols)
        , m_row_ptr(row_ptr)
        , m_col_idx(col_idx)
        , m_values(values) {}

        // Destructor
        ~CompressedDataStorage() {
            delete[] m_row_ptr;
            delete[] m_col_idx;
            delete[] m_values;
        }

    private: 
        int m_num_rows;
        int m_num_cols;
        int* m_row_ptr;
        int* m_col_idx;
        double* m_values;
};