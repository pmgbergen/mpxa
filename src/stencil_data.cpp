#include "../include/stencil_data.h"

std::shared_ptr<CompressedDataStorage<double>> stencil_to_csr(const StencilData& stencil,
                                                               int num_rows, int num_cols)
{
    // Build row_ptr: count non-zero entries per row, then prefix-sum.
    std::vector<int> row_ptr(num_rows + 1, 0);
    for (int i = 0; i < static_cast<int>(stencil.row_idx.size()); ++i)
    {
        row_ptr[stencil.row_idx[i] + 1] = static_cast<int>(stencil.col_idx[i].size());
    }
    for (int r = 0; r < num_rows; ++r)
    {
        row_ptr[r + 1] += row_ptr[r];
    }

    // Flatten col_idx and values in row order.
    const int nnz = row_ptr[num_rows];
    std::vector<int> col_idx;
    std::vector<double> values;
    col_idx.reserve(nnz);
    values.reserve(nnz);
    for (int i = 0; i < static_cast<int>(stencil.row_idx.size()); ++i)
    {
        col_idx.insert(col_idx.end(), stencil.col_idx[i].begin(), stencil.col_idx[i].end());
        values.insert(values.end(), stencil.values[i].begin(), stencil.values[i].end());
    }

    return std::make_shared<CompressedDataStorage<double>>(
        num_rows, num_cols, std::move(row_ptr), std::move(col_idx), std::move(values));
}

std::pair<std::shared_ptr<CompressedDataStorage<double>>,
          std::shared_ptr<CompressedDataStorage<double>>>
flux_stencil_to_csr(const FluxStencilData& stencil, int num_rows, int num_cols, int spatial_dim)
{
    // Build row_ptr (same sparsity for flux and vector_source).
    std::vector<int> row_ptr_flux(num_rows + 1, 0);
    for (int i = 0; i < static_cast<int>(stencil.row_idx.size()); ++i)
    {
        row_ptr_flux[stencil.row_idx[i] + 1] =
            static_cast<int>(stencil.flux_values[i].size());
    }
    for (int r = 0; r < num_rows; ++r)
    {
        row_ptr_flux[r + 1] += row_ptr_flux[r];
    }

    std::vector<int> row_ptr_vs(num_rows + 1, 0);
    for (int i = 0; i < static_cast<int>(stencil.row_idx.size()); ++i)
    {
        row_ptr_vs[stencil.row_idx[i] + 1] =
            static_cast<int>(stencil.vs_values[i].size());
    }
    for (int r = 0; r < num_rows; ++r)
    {
        row_ptr_vs[r + 1] += row_ptr_vs[r];
    }

    // Flatten flux data.
    const int nnz_flux = row_ptr_flux[num_rows];
    std::vector<int> col_idx_flux;
    std::vector<double> values_flux;
    col_idx_flux.reserve(nnz_flux);
    values_flux.reserve(nnz_flux);
    for (int i = 0; i < static_cast<int>(stencil.row_idx.size()); ++i)
    {
        col_idx_flux.insert(col_idx_flux.end(), stencil.col_idx[i].begin(),
                            stencil.col_idx[i].end());
        values_flux.insert(values_flux.end(), stencil.flux_values[i].begin(),
                           stencil.flux_values[i].end());
    }

    // Flatten vector-source data: column index for cell c component k = c*spatial_dim + k.
    const int nnz_vs = row_ptr_vs[num_rows];
    std::vector<int> col_idx_vs;
    std::vector<double> values_vs;
    col_idx_vs.reserve(nnz_vs);
    values_vs.reserve(nnz_vs);
    for (int i = 0; i < static_cast<int>(stencil.row_idx.size()); ++i)
    {
        const auto& cell_cols = stencil.col_idx[i];
        const auto& vs_vals = stencil.vs_values[i];
        for (int j = 0; j < static_cast<int>(cell_cols.size()); ++j)
        {
            for (int k = 0; k < spatial_dim; ++k)
            {
                col_idx_vs.push_back(cell_cols[j] * spatial_dim + k);
                values_vs.push_back(vs_vals[j * spatial_dim + k]);
            }
        }
    }

    auto flux_mat = std::make_shared<CompressedDataStorage<double>>(
        num_rows, num_cols, std::move(row_ptr_flux), std::move(col_idx_flux),
        std::move(values_flux));
    auto vs_mat = std::make_shared<CompressedDataStorage<double>>(
        num_rows, num_cols * spatial_dim, std::move(row_ptr_vs), std::move(col_idx_vs),
        std::move(values_vs));

    return {flux_mat, vs_mat};
}
