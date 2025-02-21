#include <compressed_storage.h>


// Constructor for an empty matrix.
CompressedDataStorage::CompressedDataStorage(int num_rows, int num_cols)
    : m_num_rows(num_rows)
    , m_num_cols(num_cols)
    , m_row_ptr(new int[num_rows + 1])
    , m_col_idx(new int[0])
    , m_values(new double[0]) {
        m_row_ptr[0] = 0;
    }

// Constructor for a matrix with indices and values given.
CompressedDataStorage::CompressedDataStorage(int num_rows, int num_cols, int* row_ptr, int* col_idx, double* values)
    : m_num_rows(num_rows)
    , m_num_cols(num_cols)
    , m_row_ptr(row_ptr)
    , m_col_idx(col_idx)
    , m_values(values) {}

// Destructor
CompressedDataStorage::~CompressedDataStorage() {
    delete[] m_row_ptr;
    delete[] m_col_idx;
    delete[] m_values;
}

int CompressedDataStorage::num_rows() {
    return m_num_rows;
}

int CompressedDataStorage::num_cols() {
    return m_num_cols;
}
