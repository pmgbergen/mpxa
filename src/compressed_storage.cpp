#include "../include/compressed_storage.h"

#include <stdexcept>

// Explicit template instantiation for CompressedDataStorage with double
template class CompressedDataStorage<int>;
template class CompressedDataStorage<double>;

// Owning constructor.
template <typename T>
CompressedDataStorage<T>::CompressedDataStorage(const int num_rows, const int num_cols,
                                                std::vector<int> row_ptr, std::vector<int> col_idx,
                                                std::vector<T> values, const bool construct_csc)
    : m_num_rows(num_rows),
      m_num_cols(num_cols),
      m_row_ptr_owned(std::move(row_ptr)),
      m_col_idx_owned(std::move(col_idx)),
      m_values_owned(std::move(values)),
      m_row_ptr(m_row_ptr_owned),
      m_col_idx(m_col_idx_owned),
      m_values(m_values_owned),
      m_csc_constructed(construct_csc)
{
    if (m_row_ptr.size() != static_cast<std::size_t>(m_num_rows + 1))
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
        build_csc();
    }
}

// Non-owning constructor.
template <typename T>
CompressedDataStorage<T>::CompressedDataStorage(const int num_rows, const int num_cols,
                                                const int* row_ptr, std::size_t row_ptr_size,
                                                const int* col_idx, std::size_t col_idx_size,
                                                const T* values, std::size_t values_size,
                                                const bool construct_csc)
    : m_num_rows(num_rows),
      m_num_cols(num_cols),
      m_row_ptr(row_ptr, row_ptr_size),
      m_col_idx(col_idx, col_idx_size),
      m_values(values, values_size),
      m_csc_constructed(construct_csc)
{
    if (m_row_ptr.size() != static_cast<std::size_t>(m_num_rows + 1))
    {
        throw std::invalid_argument("Row pointer size does not match number of rows.");
    }
    if (m_col_idx.size() != m_values.size())
    {
        throw std::invalid_argument("Column index and values size do not match.");
    }
    if (m_csc_constructed)
    {
        build_csc();
    }
}

// from_csc factory: accept CSC arrays (non-owning) and build CSR internally.
template <typename T>
std::shared_ptr<CompressedDataStorage<T>> CompressedDataStorage<T>::from_csc(
    int num_rows, int num_cols,
    const int* col_ptr, std::size_t col_ptr_size,
    const int* row_idx, std::size_t row_idx_size,
    const T* values, std::size_t values_size)
{
    const std::span<const int> csc_col_ptr(col_ptr, col_ptr_size);
    const std::span<const int> csc_row_idx(row_idx, row_idx_size);
    const std::span<const T> csc_values(values, values_size);

    const int nnz = static_cast<int>(row_idx_size);

    // Build CSR from CSC.
    std::vector<int> csr_row_ptr(num_rows + 1, 0);
    std::vector<int> csr_col_idx(nnz);
    std::vector<T> csr_values(nnz);

    // Count non-zeros per row.
    for (int i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr[csc_row_idx[i] + 1];
    }
    // Cumulative sum.
    for (int row = 0; row < num_rows; ++row)
    {
        csr_row_ptr[row + 1] += csr_row_ptr[row];
    }
    // Fill col indices and values.
    std::vector<int> counter = csr_row_ptr;
    for (int col = 0; col < num_cols; ++col)
    {
        for (int idx = csc_col_ptr[col]; idx < csc_col_ptr[col + 1]; ++idx)
        {
            const int row = csc_row_idx[idx];
            const int dest = counter[row]++;
            csr_col_idx[dest] = col;
            csr_values[dest] = csc_values[idx];
        }
    }

    // Construct with owned CSR data; CSC views set manually below.
    auto obj = std::make_shared<CompressedDataStorage<T>>(
        num_rows, num_cols,
        std::move(csr_row_ptr), std::move(csr_col_idx), std::move(csr_values),
        false);

    // Wire in the non-owning CSC views from the original input arrays.
    obj->m_col_ptr_view = csc_col_ptr;
    obj->m_row_idx_view = csc_row_idx;
    obj->m_values_csc_view = csc_values;
    obj->m_csc_constructed = true;

    return obj;
}

template <typename T>
void CompressedDataStorage<T>::build_csc()
{
    const int nnz = static_cast<int>(m_col_idx.size());

    m_col_ptr.assign(m_num_cols + 1, 0);
    m_row_idx.resize(nnz);
    m_values_csc.resize(nnz);

    for (int i = 0; i < nnz; ++i)
    {
        ++m_col_ptr[m_col_idx[i] + 1];
    }
    for (int col = 0; col < m_num_cols; ++col)
    {
        m_col_ptr[col + 1] += m_col_ptr[col];
    }

    std::vector<int> counter = m_col_ptr;
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

    m_col_ptr_view = m_col_ptr;
    m_row_idx_view = m_row_idx;
    m_values_csc_view = m_values_csc;
}

template <typename T>
int CompressedDataStorage<T>::num_rows() const
{
    return m_num_rows;
}

template <typename T>
int CompressedDataStorage<T>::num_cols() const
{
    return m_num_cols;
}

template <typename T>
std::span<const int> CompressedDataStorage<T>::row_ptr() const
{
    return m_row_ptr;
}

template <typename T>
std::span<const int> CompressedDataStorage<T>::col_idx() const
{
    return m_col_idx;
}

template <typename T>
std::span<const T> CompressedDataStorage<T>::data() const
{
    return m_values;
}

template <typename T>
std::span<const int> CompressedDataStorage<T>::cols_in_row(int row) const
{
    const int start = m_row_ptr[row];
    const int size = m_row_ptr[row + 1] - start;
    return std::span<const int>(m_col_idx.data() + start, size);
}

template <typename T>
std::vector<int> CompressedDataStorage<T>::rows_in_col(int col) const
{
    if (m_csc_constructed)
    {
        const int size = m_col_ptr_view[col + 1] - m_col_ptr_view[col];
        std::vector<int> rows(size);
        for (int i = 0; i < size; i++)
        {
            rows[i] = m_row_idx_view[m_col_ptr_view[col] + i];
        }
        return rows;
    }

    std::vector<int> rows;
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
    return rows;
}

template <typename T>
std::vector<T> CompressedDataStorage<T>::values() const
{
    return std::vector<T>(m_values.begin(), m_values.end());
}

template <typename T>
T CompressedDataStorage<T>::value(const int row, const int col) const
{
    for (int i = m_row_ptr[row]; i < m_row_ptr[row + 1]; i++)
    {
        if (m_col_idx[i] == col)
        {
            return m_values[i];
        }
    }
    return 0;
}
