#include <vector>
#include "compressed_storage.h"

// Constructor for an empty matrix.
template <typename T>
CompressedDataStorage<T>::CompressedDataStorage(int num_rows, int num_cols)
    : m_num_rows(num_rows)
    , m_num_cols(num_cols)
    , m_row_ptr(new int[num_rows + 1])
    , m_col_idx(new int[0])
    , m_values(new T[0]) {
        m_row_ptr[0] = 0;
    }

// Constructor for a matrix with indices and values given.
template <typename T>
CompressedDataStorage<T>::CompressedDataStorage(int num_rows, int num_cols, int* row_ptr, int* col_idx, T* values)
    : m_num_rows(num_rows)
    , m_num_cols(num_cols)
    , m_row_ptr(row_ptr)
    , m_col_idx(col_idx)
    , m_values(values) {}

// Destructor
template <typename T>
CompressedDataStorage<T>::~CompressedDataStorage() {
    delete[] m_row_ptr;
    delete[] m_col_idx;
    delete[] m_values;
}

template <typename T>
const int CompressedDataStorage<T>::num_rows() {
    return m_num_rows;
}

template <typename T>
const int CompressedDataStorage<T>::num_cols() {
    return m_num_cols;
}

template <typename T>
const int* CompressedDataStorage<T>::cols_in_row(int row) {
    const int size = m_row_ptr[row + 1] - m_row_ptr[row];
    const int* cols = new int[size];
    for (int i = 0; i < size; i++) {
        cols[i] = m_col_idx[m_row_ptr[row] + i];
    }
    return cols;
}

template <typename T>
const int* CompressedDataStorage<T>::rows_in_col(int col) {
    std::vector<int> rows;
    // Loop over all rows, find the column index in the row. If the column of the row is the same as the input column, add the row index to the list.
    for (int i = 0; i < m_num_rows; i++) {
        for (int j = m_row_ptr[i]; j < m_row_ptr[i + 1]; j++) {
            if (m_col_idx[j] == col) {
                rows.push_back(i);
            }
        }
    }
    // Convert the list to an array and return it.
    int* result = new int[rows.size()];
    std::copy(rows.begin(), rows.end(), result);
    return result;
}

template <typename T>
const T* CompressedDataStorage<T>::values() {
    return m_values;
}

template <typename T>
const T* CompressedDataStorage<T>::values_in_row(int row) {
    const int size = m_row_ptr[row + 1] - m_row_ptr[row];
    const T* values = new T[size];
    for (int i = 0; i < size; i++) {
        values[i] = m_values[m_row_ptr[row] + i];
    }
    return values;
}