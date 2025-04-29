#include "../include/compressed_storage.h"

// Explicit template instantiation for CompressedDataStorage with double
template class CompressedDataStorage<int>;
template class CompressedDataStorage<double>;

// Constructor for a matrix with indices and values given.
template <typename T>
CompressedDataStorage<T>::CompressedDataStorage(const int num_rows, const int num_cols,
                                                const std::vector<int>& row_ptr,
                                                const std::vector<int>& col_idx,
                                                const std::vector<T>& values)
    : m_num_rows(num_rows),
      m_num_cols(num_cols),
      m_row_ptr(row_ptr),
      m_col_idx(col_idx),
      m_values(values)
{
}

// Destructor
template <typename T>
CompressedDataStorage<T>::~CompressedDataStorage()
{
    // No need to manually delete arrays, std::vector handles it.
}

template <typename T>
const int CompressedDataStorage<T>::num_rows()
{
    return m_num_rows;
}

template <typename T>
const int CompressedDataStorage<T>::num_cols()
{
    return m_num_cols;
}

template <typename T>
std::vector<int> CompressedDataStorage<T>::cols_in_row(int row)
{
    const int size = m_row_ptr[row + 1] - m_row_ptr[row];
    std::vector<int> cols(size);
    for (int i = 0; i < size; i++)
    {
        cols[i] = m_col_idx[m_row_ptr[row] + i];
    }
    return cols;
}

template <typename T>
std::vector<int> CompressedDataStorage<T>::rows_in_col(int col)
{
    std::vector<int> rows;
    // Loop over all rows, find the column index in the row. If the column of the row is
    // the same as the input column, add the row index to the list.
    for (int i = 0; i < m_num_rows; i++)
    {
        for (int j = m_row_ptr[i]; j < m_row_ptr[i + 1]; j++)
        {
            if (m_col_idx[j] == col)
            {
                rows.push_back(i);
            }
        }
    }
    // Convert the list to an array and return it.
    return rows;
}

template <typename T>
std::vector<T> CompressedDataStorage<T>::values()
{
    return m_values;
}

template <typename T>
T CompressedDataStorage<T>::value(const int row, const int col)
{
    // Loop over all values in the row, find the column index in the row. If the column of the
    // row is the same as the input column, return the value.
    for (int i = m_row_ptr[row]; i < m_row_ptr[row + 1]; i++)
    {
        if (m_col_idx[i] == col)
        {
            return m_values[i];
        }
    }
    // If the column is not found, return 0.
    return 0;
}
