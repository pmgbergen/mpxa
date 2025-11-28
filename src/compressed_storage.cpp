#include "../include/compressed_storage.h"

#include <iostream>

// Explicit template instantiation for CompressedDataStorage with double
template class CompressedDataStorage<int>;
template class CompressedDataStorage<double>;

// Constructor for a matrix with indices and values given.
template <typename T>
CompressedDataStorage<T>::CompressedDataStorage(const int num_rows, const int num_cols,
                                                const std::vector<int>& row_ptr,
                                                const std::vector<int>& col_idx,
                                                const std::vector<T>& values,
                                                const bool construct_csc)
    : m_num_rows(num_rows),
      m_num_cols(num_cols),
      m_row_ptr(row_ptr),
      m_col_idx(col_idx),
      m_values(values),
      m_csc_constructed(construct_csc)
{
    // Check if the sizes of the vectors are consistent with the number of rows and columns.
    if (m_row_ptr.size() != m_num_rows + 1)
    {
        throw std::invalid_argument("Row pointer size does not match number of rows.");
    }
    if (m_col_idx.size() != m_values.size())
    {
        throw std::invalid_argument("Column index and values size do not match.");
    }
    for (int i = 0; i < m_num_rows; i++)
    {
        if (m_row_ptr[i] > m_row_ptr[i + 1])
        {
            throw std::invalid_argument("Row pointer is not sorted.");
        }
    }
    if (m_csc_constructed)
    {
        m_col_ptr.assign(m_num_cols + 1, 0);
        m_row_idx.resize(m_col_idx.size());
        m_values_csc.resize(m_values.size());

        const int nnz = m_col_idx.size();

        // Step 1: Count non-zeros per column.
        for (int i = 0; i < nnz; ++i)
        {
            const int col = m_col_idx[i];
            ++m_col_ptr[col + 1];
        }

        // Step 2: Cumulative sum to get m_col_ptr
        for (int col = 0; col < m_num_cols; ++col)
        {
            m_col_ptr[col + 1] += m_col_ptr[col];
        }

        // Step 3: Fill m_row_idx and m_values_csc
        std::vector<int> counter = m_col_ptr;  // Will track insert positions

        for (int row = 0; row < m_num_rows; ++row)
        {
            for (int idx = m_row_ptr[row]; idx < m_row_ptr[row + 1]; ++idx)
            {
                const int col = m_col_idx[idx];
                const int dest_pos = counter[col]++;

                m_row_idx[dest_pos] = row;
                m_values_csc[dest_pos] = m_values[idx];
            }
        }
    }
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
std::span<const int> CompressedDataStorage<T>::cols_in_row(int row)
{
    const int start = m_row_ptr[row];
    const int size = m_row_ptr[row + 1] - start;
    return std::span<const int>(&m_col_idx[start], size);
}

template <typename T>
std::vector<int> CompressedDataStorage<T>::rows_in_col(int col)
{
    if (m_csc_constructed)
    {
        const int size = m_col_ptr[col + 1] - m_col_ptr[col];
        std::vector<int> rows(size);
        for (int i = 0; i < size; i++)
        {
            rows[i] = m_row_idx[m_col_ptr[col] + i];
        }
        return rows;
    }
    // If CSC format is not constructed, we need to search through the entire matrix.

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
const std::vector<int>& CompressedDataStorage<T>::row_ptr()
{
    return m_row_ptr;
}

template <typename T>
const std::vector<int>& CompressedDataStorage<T>::col_idx() const
{
    return m_col_idx;
}

template <typename T>
const std::vector<T>& CompressedDataStorage<T>::data()
{
    return m_values;
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
