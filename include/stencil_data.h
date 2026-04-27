#ifndef STENCIL_DATA_H
#define STENCIL_DATA_H

// Sparse-matrix stencil data structures shared by both the MPFA and TPFA
// discretizations.  Each structure stores data in a jagged-array (COO-like)
// format: one outer entry per contribution, each holding vectors of column
// indices and values.  The MPFA discretization may produce multiple
// contributions for the same matrix row (one per interaction region); the TPFA
// discretization produces exactly one contribution per row.

#include <memory>
#include <utility>
#include <vector>

#include "compressed_storage.h"

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// Row-index / column-index / values triplet for building one sparse matrix.
struct StencilData
{
    std::vector<int> row_idx;
    std::vector<std::vector<int>> col_idx;
    std::vector<std::vector<double>> values;

    void reserve(int capacity)
    {
        row_idx.reserve(capacity);
        col_idx.reserve(capacity);
        values.reserve(capacity);
    }
};

// Stencil data for the flux matrix and its associated vector source term.
// The two share the same sparsity pattern (row_idx and col_idx).
// vs_values stores SPATIAL_DIM values per cell column entry; the vector-source
// CSR column index for cell c in position j is col_idx[i][j] * SPATIAL_DIM + k.
struct FluxStencilData
{
    std::vector<int> row_idx;
    std::vector<std::vector<int>> col_idx;
    std::vector<std::vector<double>> flux_values;
    std::vector<std::vector<double>> vs_values;
};

// Stencil data for the four boundary discretization matrices, grouped by matrix.
struct BoundaryStencilData
{
    StencilData bound_flux;
    StencilData pressure_cell;
    StencilData pressure_face;
    StencilData vector_source;
};

// ---------------------------------------------------------------------------
// CSR conversion helpers for sequential-row stencils
// ---------------------------------------------------------------------------

// Convert a StencilData to a CSR matrix.
// Requirement: row_idx entries are non-decreasing and each row appears at most
// once (the TPFA case).  Rows absent from row_idx produce empty rows.
std::shared_ptr<CompressedDataStorage<double>> stencil_to_csr(const StencilData& stencil,
                                                               int num_rows, int num_cols);

// Convert a FluxStencilData to a (flux, vector_source) pair of CSR matrices.
// Same requirement as stencil_to_csr: non-decreasing, non-repeated row_idx.
// SPATIAL_DIM is the number of spatial components (columns per cell in vs).
std::pair<std::shared_ptr<CompressedDataStorage<double>>,
          std::shared_ptr<CompressedDataStorage<double>>>
flux_stencil_to_csr(const FluxStencilData& stencil, int num_rows, int num_cols,
                    int spatial_dim);

#endif  // STENCIL_DATA_H
