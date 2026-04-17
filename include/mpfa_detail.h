#ifndef MPFA_DETAIL_H
#define MPFA_DETAIL_H

// Internal helpers for the MPFA discretisation, exposed in a named namespace so
// that unit tests can access them directly without going through the public mpfa()
// entry point.

#include <Eigen/Dense>
#include <array>
#include <map>
#include <memory>
#include <vector>

#include "compressed_storage.h"
#include "discr.h"
#include "grid.h"
#include "stencil_data.h"
#include "tensor.h"

namespace mpfa_detail
{

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// Scratch information produced by compute_row_sorting, shared between both
// CSR-matrix builders.
struct RowSortingInfo
{
    std::vector<int> sorted_row_indices;
    std::vector<int> num_row_occurrences;
    std::vector<int> col_index_sizes;
};

// Local Eigen matrices for one interaction region, bundled to reduce parameter counts.
struct LocalBalanceMatrices
{
    Eigen::MatrixXd balance_cells;
    Eigen::MatrixXd balance_faces;
    Eigen::MatrixXd flux_cells;
    Eigen::MatrixXd flux_faces;
    Eigen::MatrixXd nK_matrix;
    Eigen::MatrixXd nK_one_sided;
    std::map<int, std::vector<std::array<double, 3>>> basis_map;
};

// ---------------------------------------------------------------------------
// Pure utility functions
// ---------------------------------------------------------------------------

// Compute n·K / num_nodes for a face, returning a 3-element array.
const std::array<double, 3> nK(const std::array<double, 3>& face_normal,
                               const SecondOrderTensor& tensor, int cell_ind,
                               int num_nodes_of_face);

// Pad or truncate a std::vector<double> into a 3-element array (zeros for missing entries).
std::array<double, 3> to_array3(const std::vector<double>& v);

// Compute the dot product of nk_val with each basis function vector.
std::vector<double> nKgrad(const std::array<double, 3>& nk_val,
                           const std::vector<std::array<double, 3>>& basis_functions);

// Compute the projection of (face_center - cell_center) onto each basis function.
std::vector<double> p_diff(const std::array<double, 3>& face_center,
                           const std::array<double, 3>& cell_center,
                           const std::vector<std::array<double, 3>>& basis_functions);

// ---------------------------------------------------------------------------
// Grid-topology helpers
// ---------------------------------------------------------------------------

std::vector<int> count_nodes_of_faces(const Grid& grid);
std::vector<int> count_faces_of_cells(const Grid& grid);

// Return true if every cell has exactly dim+1 faces (i.e., the grid is simplicial).
bool check_if_simplex(const std::vector<int>& num_faces_of_cell, int dim);

// ---------------------------------------------------------------------------
// CSR matrix helpers
// ---------------------------------------------------------------------------

// Compute sorted row indices, per-row occurrence counts, and per-entry column sizes.
RowSortingInfo compute_row_sorting(const std::vector<int>& row_indices,
                                   const std::vector<std::vector<int>>& col_indices,
                                   int num_rows);

// Build a CSR sparse matrix, accumulating values for repeated (row, col) pairs.
std::shared_ptr<CompressedDataStorage<double>> create_csr_matrix(
    const std::vector<int>& row_indices, const std::vector<std::vector<int>>& col_indices,
    const std::vector<std::vector<double>>& data_values, int num_rows, int num_cols,
    int tot_num_transmissibilities);

// ---------------------------------------------------------------------------
// Interaction-region matrix helpers
// ---------------------------------------------------------------------------

// Create zero-initialised local balance matrices for an interaction region.
LocalBalanceMatrices make_local_balance_matrices(int num_faces, int num_cells);

// Initialise a FluxStencilData with pre-reserved capacity.
FluxStencilData init_flux_stencil(int num_faces, int nodes_per_face, int dim);

// Initialise a BoundaryStencilData with pre-reserved capacity.
BoundaryStencilData init_boundary_stencil(int num_faces, int num_bound_faces,
                                          int nodes_per_face, int dim);

}  // namespace mpfa_detail

#endif  // MPFA_DETAIL_H
