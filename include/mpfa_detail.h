#ifndef MPFA_DETAIL_H
#define MPFA_DETAIL_H

// Internal helpers for the MPFA discretisation, exposed in a named namespace so
// that unit tests can access them directly without going through the public mpfa()
// entry point.

#include <Eigen/Dense>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "compressed_storage.h"
#include "discr.h"
#include "grid.h"
#include "multipoint_common.h"
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

// Derived local flux matrices computed from LocalBalanceMatrices.
struct LocalFluxMatrices
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> balance_faces_inv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bound_flux;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flux;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vector_source_cell;
};

// Cached geometry for one interaction region.
struct InteractionRegionGeometry
{
    std::vector<std::array<double, 3>> cell_centers;
    std::vector<std::array<double, 3>> face_centers;
    std::vector<std::array<double, 3>> face_normals;
};

// Classification of boundary faces within an interaction region.
struct BoundaryFaceClassification
{
    // types[loc_face_ind] holds the BC type for boundary faces, nullopt for internal faces.
    std::vector<std::optional<BoundaryCondition>> types;
    // (local face index, global face index) pairs for all boundary faces.
    std::vector<std::pair<int, int>> boundary_face_map;
};

// Read-only discretisation inputs shared across the entire grid.
struct DiscrContext
{
    const Grid& grid;
    const SecondOrderTensor& tensor;
    const std::vector<int>& num_nodes_of_face;
    const std::unordered_map<int, BoundaryCondition>& bc_map;
};

// Bundles all read-only inputs needed by compute_continuity_points_for_cell,
// dropping the parameter count from 7 to 2.
struct ContinuityPointContext
{
    const InteractionRegion& region;
    const InteractionRegionGeometry& geom;
    const std::array<double, 3>& node_coord;
    const std::vector<std::optional<BoundaryCondition>>& bc_types;
    bool is_simplex;
    int dim;
};

// Bundles all read-only inputs needed by fill_cell_contributions,
// dropping the parameter count from 6 to 2.
struct FaceCellContribContext
{
    const InteractionRegion& region;
    const DiscrContext& discr_ctx;
    const InteractionRegionGeometry& geom;
    const std::vector<std::optional<BoundaryCondition>>& bc_types;
    const std::vector<std::array<double, 3>>& basis_functions;
    int loc_cell_ind;
};

// Bundles inputs for the Neumann pressure reconstruction in accumulate_boundary_data.
// One instance is created per interaction region and reused for all Neumann faces.
// All Eigen matrix members use RowMajor storage to match LocalFluxMatrices exactly and
// avoid any implicit conversion (which would create a temporary and a dangling reference).
using RowMajorXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
struct NeumannReconstructionContext
{
    const InteractionRegion& region;
    const InteractionRegionGeometry& geom;
    const LocalBalanceMatrices& matrices;
    const RowMajorXd& balance_faces_inv;
    const RowMajorXd& bound_vector_source_matrix;
    const RowMajorXd& face_pressure_from_cells;
    const std::vector<std::optional<BoundaryCondition>>& bc_types;
    const std::vector<int>& glob_indices_iareg_faces;
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
// Interaction-region geometry and classification helpers
// ---------------------------------------------------------------------------

InteractionRegionGeometry compute_interaction_region_geometry(const InteractionRegion& region,
                                                               const Grid& grid);

BoundaryFaceClassification classify_boundary_faces(
    const InteractionRegion& region,
    const std::unordered_map<int, BoundaryCondition>& bc_map);

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

// Compute derived local flux matrices from balance matrices and boundary conditions.
LocalFluxMatrices compute_local_flux(
    const LocalBalanceMatrices& matrices,
    const std::vector<std::optional<BoundaryCondition>>& loc_boundary_faces_type);

// Initialise a FluxStencilData with pre-reserved capacity.
FluxStencilData init_flux_stencil(int num_faces, int nodes_per_face, int dim);

// Initialise a BoundaryStencilData with pre-reserved capacity.
BoundaryStencilData init_boundary_stencil(int num_faces, int num_bound_faces,
                                          int nodes_per_face, int dim);

// ---------------------------------------------------------------------------
// Per-cell contribution helpers
// ---------------------------------------------------------------------------

// Compute continuity points for one cell within an interaction region.
std::vector<std::array<double, 3>> compute_continuity_points_for_cell(
    int loc_cell_ind, const ContinuityPointContext& ctx);

// Fill balance and flux matrix contributions for one cell within an interaction region.
void fill_cell_contributions(const FaceCellContribContext& ctx, LocalBalanceMatrices& matrices);

// ---------------------------------------------------------------------------
// Boundary stencil accumulation helpers
// ---------------------------------------------------------------------------

// Build bound_flux entries for all faces in the interaction region.
void accumulate_bound_flux_entries(const InteractionRegion& region,
                                   const LocalFluxMatrices& local_flux,
                                   const BoundaryFaceClassification& bc_class,
                                   const std::vector<int>& num_nodes_of_face,
                                   BoundaryStencilData& out);

// Add a unit pressure_face entry for a Dirichlet boundary face.
void accumulate_dirichlet_pressure_face(int glob_face_ind, double inv_num_nodes,
                                        BoundaryStencilData& out);

// Add pressure_cell, pressure_face, and vector_source entries for one Neumann face.
void accumulate_neumann_pressure_reconstruction(const NeumannReconstructionContext& ctx,
                                                int loc_face_ind, int glob_face_ind,
                                                double inv_num_nodes, BoundaryStencilData& out);

// Accumulate all boundary stencil data for one interaction region.
void accumulate_boundary_data(const InteractionRegion& interaction_region,
                              const LocalFluxMatrices& local_flux,
                              const BoundaryFaceClassification& bc_class,
                              const InteractionRegionGeometry& geom,
                              const LocalBalanceMatrices& matrices,
                              const std::vector<int>& num_nodes_of_face,
                              BoundaryStencilData& out);

// ---------------------------------------------------------------------------
// Main loop helper
// ---------------------------------------------------------------------------

// Process one interaction region: compute local matrices, accumulate flux and
// boundary stencil data.  This function contains the per-node loop body of
// mpfa() and is ready for OpenMP parallelisation once thread-safe accumulators
// are introduced.
void process_interaction_region(int node_ind, const DiscrContext& ctx,
                                 const std::vector<int>& num_faces_of_cell,
                                 FluxStencilData& flux_out, BoundaryStencilData& bound_out,
                                 int& tot_num_trm);

}  // namespace mpfa_detail

#endif  // MPFA_DETAIL_H
