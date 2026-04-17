#include <Eigen/Dense>
#include <array>
#include <map>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "../include/discr.h"
#include "../include/mpfa_detail.h"
#include "../include/multipoint_common.h"

using Eigen::MatrixXd;

namespace mpfa_detail
{

const std::array<double, 3> nK(const std::array<double, 3>& face_normal,
                               const SecondOrderTensor& tensor, const int cell_ind,
                               const int num_nodes_of_face)
{
    // Compute the product between the normal vector, the tensor, and the cell-face vector.

    constexpr int dim = 3;  // face_normal.size();
    std::array<double, dim> result = {0.0, 0.0, 0.0};

    // Compute inverse ratio to limit the number of divisions.
    const double num_nodes_of_face_inv = 1.0 / num_nodes_of_face;

    if (tensor.is_isotropic())
    {
        for (int i{0}; i < dim; ++i)
        {
            result[i] = -face_normal[i] * tensor.isotropic_data(cell_ind) * num_nodes_of_face_inv;
        }
    }
    else if (tensor.is_diagonal())
    {
        auto diag = tensor.diagonal_data(cell_ind);
        for (int i{0}; i < dim; ++i)
        {
            result[i] = -face_normal[i] * diag[i] * num_nodes_of_face_inv;
        }
    }
    else
    {
        auto full_data = tensor.full_data(cell_ind);
        const double xx = full_data[0];
        const double yy = full_data[1];
        const double zz = full_data[2];
        const double xy = full_data[3];
        const double xz = full_data[4];
        const double yz = full_data[5];
        result[0] = -num_nodes_of_face_inv *
                    (face_normal[0] * xx + face_normal[1] * xy + face_normal[2] * xz);
        result[1] = -num_nodes_of_face_inv *
                    (face_normal[0] * xy + face_normal[1] * yy + face_normal[2] * yz);
        result[2] = -num_nodes_of_face_inv *
                    (face_normal[0] * xz + face_normal[1] * yz + face_normal[2] * zz);
    }
    return result;
}

// Helper function to get cell center coordinates for all cells in an interaction region
// Convert a std::vector<double> to a 3-element array, padding with zeros.
std::array<double, 3> to_array3(const std::vector<double>& v)
{
    std::array<double, 3> arr{0.0, 0.0, 0.0};
    for (size_t i = 0; i < std::min(v.size(), arr.size()); ++i)
    {
        arr[i] = v[i];
    }
    return arr;
}

// Cell centers, face centers, and face normals for all cells/faces in an interaction region.
struct InteractionRegionGeometry
{
    std::vector<std::array<double, 3>> cell_centers;
    std::vector<std::array<double, 3>> face_centers;
    std::vector<std::array<double, 3>> face_normals;
};

InteractionRegionGeometry compute_interaction_region_geometry(
    const InteractionRegion& region, const Grid& grid)
{
    InteractionRegionGeometry geom;
    geom.cell_centers.reserve(region.cells().size());
    for (const int cell_ind : region.cells())
    {
        geom.cell_centers.push_back(to_array3(grid.cell_center(cell_ind)));
    }
    geom.face_centers.reserve(region.faces().size());
    geom.face_normals.reserve(region.faces().size());
    for (const auto& pair : region.faces())
    {
        geom.face_centers.push_back(to_array3(grid.face_center(pair.first)));
        geom.face_normals.push_back(to_array3(grid.face_normal(pair.first)));
    }
    return geom;
}

std::vector<double> nKgrad(const std::array<double, 3>& nK,
                           const std::vector<std::array<double, 3>>& basis_functions)
{
    // Compute the gradient of the nK expression at the given face index.
    std::vector<double> grad(basis_functions.size(), 0.0);
    for (size_t i = 0; i < basis_functions.size(); ++i)
    {
        for (size_t j = 0; j < basis_functions[i].size(); ++j)
        {
            grad[i] += nK[j] * basis_functions[i][j];
        }
    }
    return grad;
}

std::vector<double> p_diff(const std::array<double, 3>& face_center,
                           const std::array<double, 3>& cell_center,
                           const std::vector<std::array<double, 3>>& basis_functions)
{
    std::array<double, 3> dist = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < face_center.size(); ++i)
    {
        dist[i] = face_center[i] - cell_center[i];
    }
    std::vector<double> diff(basis_functions.size(), 0.0);
    for (size_t i = 0; i < basis_functions.size(); ++i)
    {
        for (size_t j = 0; j < basis_functions[i].size(); ++j)
        {
            diff[i] += dist[j] * basis_functions[i][j];
        }
    }

    return diff;
}

std::vector<int> count_nodes_of_faces(const Grid& grid)
{
    // Count the number of nodes for each face in the grid.
    std::vector<int> num_nodes_of_face(grid.num_faces(), 0);

    const auto& face_nodes = grid.face_nodes();

    const auto& col_idx = face_nodes.col_idx();

    for (const int face_ind : col_idx)
    {
        ++num_nodes_of_face[face_ind];
    }

    return num_nodes_of_face;
}

std::vector<int> count_faces_of_cells(const Grid& grid)
{
    // Count the number of faces for each cell in the grid.
    std::vector<int> num_faces_of_cell(grid.num_cells(), 0);
    const auto& cell_faces = grid.cell_faces();
    const auto& col_idx = cell_faces.col_idx();
    for (const int face_ind : col_idx)
    {
        num_faces_of_cell[face_ind]++;
    }
    return num_faces_of_cell;
}

// Returns true if all cells in the grid have exactly dim+1 faces (i.e., the grid is simplicial).
bool check_if_simplex(const std::vector<int>& num_faces_of_cell, int dim)
{
    for (const int num : num_faces_of_cell)
    {
        if (num != (dim + 1))
        {
            return false;
        }
    }
    return true;
}

struct PairHash
{
    std::size_t operator()(const std::pair<int, int>& p) const noexcept
    {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// Compute sorted row indices, per-row occurrence counts, and per-entry column sizes.
// This is common setup for both CSR matrix builders.
RowSortingInfo compute_row_sorting(const std::vector<int>& row_indices,
                                   const std::vector<std::vector<int>>& col_indices,
                                   int num_rows)
{
    RowSortingInfo info;
    info.sorted_row_indices.resize(row_indices.size());
    std::iota(info.sorted_row_indices.begin(), info.sorted_row_indices.end(), 0);
    std::sort(info.sorted_row_indices.begin(), info.sorted_row_indices.end(),
              [&row_indices](int i1, int i2) { return row_indices[i1] < row_indices[i2]; });

    info.num_row_occurrences.assign(num_rows, 0);
    for (const int row_ind : row_indices)
    {
        ++info.num_row_occurrences[row_ind];
    }
    info.col_index_sizes.reserve(col_indices.size());
    for (const auto& vec : col_indices)
    {
        info.col_index_sizes.push_back(static_cast<int>(vec.size()));
    }
    return info;
}

// Helper function to create a compressed sparse row (CSR) matrix from vectors.
std::shared_ptr<CompressedDataStorage<double>> create_csr_matrix(
    const std::vector<int>& row_indices, const std::vector<std::vector<int>>& col_indices,
    const std::vector<std::vector<double>>& data_values, const int num_rows, const int num_cols,
    const int tot_num_transmissibilities)
{
    const auto sorting = compute_row_sorting(row_indices, col_indices, num_rows);
    const auto& sorted_row_indices = sorting.sorted_row_indices;
    const auto& num_row_occurrences = sorting.num_row_occurrences;
    const auto& col_index_sizes = sorting.col_index_sizes;

    std::vector<int> row_ptr;
    row_ptr.reserve(num_rows + 1);
    row_ptr.push_back(0);
    std::vector<int> col_idx;
    col_idx.reserve(tot_num_transmissibilities);
    std::vector<double> values;
    values.reserve(tot_num_transmissibilities);

    // Local storage for the sorted column indices and data values for each row. Will be
    // erased for each iteration.
    std::vector<int> sorted_col_indices;
    std::vector<double> sorted_data_values;
    std::vector<int> this_row_col_indices;
    std::vector<double> this_row_data;

    int current_ind = 0;
    for (int row_ind = 0; row_ind < static_cast<int>(num_row_occurrences.size()); ++row_ind)
    {
        if (num_row_occurrences[row_ind] == 0)
        {
            // No entries for this row, just copy the previous row pointer.
            row_ptr.push_back(row_ptr.back());
            continue;
        }
        // Create a vector of the local sorted indices for this row.
        std::vector<int> loc_sorted_indices(num_row_occurrences[row_ind]);
        for (int i = 0; i < num_row_occurrences[row_ind]; ++i)
        {
            loc_sorted_indices[i] = sorted_row_indices[current_ind + i];
        }
        current_ind += num_row_occurrences[row_ind];

        // Determine the total number of column indices for this row. This may contained
        // repeated indices, but we will ignore that for now.
        int num_data_this_row = 0;
        for (int i = 0; i < num_row_occurrences[row_ind]; ++i)
        {
            num_data_this_row += col_index_sizes[loc_sorted_indices[i]];
        }

        // Gather all column indices and data values for this row. This will gather
        // contributions from several local calculations (several interaction regions).
        this_row_col_indices.reserve(num_data_this_row);
        this_row_data.reserve(num_data_this_row);
        for (int i = 0; i < num_row_occurrences[row_ind]; ++i)
        {
            // Get the local column indices and data values for this local calculation.
            // This may be a performance bottleneck (risk of page faults), but
            // considering the data is unstructured, we need to take that hit somewhere.
            const auto& loc_col_indices = col_indices[loc_sorted_indices[i]];
            const auto& loc_data_values = data_values[loc_sorted_indices[i]];
            this_row_col_indices.insert(this_row_col_indices.end(), loc_col_indices.begin(),
                                        loc_col_indices.end());
            this_row_data.insert(this_row_data.end(), loc_data_values.begin(),
                                 loc_data_values.end());
        }

        if (this_row_col_indices.size() == 0)
        {
            // No entries for this row, just copy the previous row pointer.
            row_ptr.push_back(col_idx.size());
            continue;
        }

        // Now we need to sort the column indices and data values according to column
        std::vector<int> sorting_col_indices(this_row_col_indices.size());
        std::iota(sorting_col_indices.begin(), sorting_col_indices.end(), 0);
        std::sort(sorting_col_indices.begin(), sorting_col_indices.end(),
                  [&this_row_col_indices](int a, int b)
                  { return this_row_col_indices[a] < this_row_col_indices[b]; });

        // Create the sorted column indices and data values. These are used to
        // accumulate values for repeated column indices.
        sorted_col_indices.reserve(this_row_col_indices.size());
        sorted_data_values.reserve(this_row_data.size());

        int prev_col = this_row_col_indices[sorting_col_indices[0]];
        double accum_data = 0.0;

        for (int i : sorting_col_indices)
        {
            if (this_row_col_indices[i] == prev_col)
            {
                // Accumulate data for repeated column indices.
                accum_data += this_row_data[i];
            }
            else
            {
                // Store the accumulated value and reset for the new column index.
                sorted_col_indices.push_back(prev_col);
                sorted_data_values.push_back(accum_data);

                prev_col = this_row_col_indices[i];
                accum_data = this_row_data[i];
            }
        }

        // Add the last accumulated value.
        sorted_col_indices.push_back(prev_col);
        sorted_data_values.push_back(accum_data);

        // Append the sorted column indices and data values to the global storage.
        col_idx.insert(col_idx.end(), sorted_col_indices.begin(), sorted_col_indices.end());
        values.insert(values.end(), sorted_data_values.begin(), sorted_data_values.end());
        row_ptr.push_back(col_idx.size());
        // Clear the local storage for the next iteration.
        this_row_col_indices.clear();
        this_row_data.clear();
        sorted_col_indices.clear();
        sorted_data_values.clear();
    }

    // Create the compressed data storage for the flux.
    // EK note to self: The cost of the matrix construction is negligible here.
    auto matrix = std::make_shared<CompressedDataStorage<double>>(
        num_rows, num_cols, std::move(row_ptr), std::move(col_idx), std::move(values));
    return matrix;
}

// Helper function to create a compressed sparse row (CSR) matrix from vectors for both
// flux and vector source terms. This is a specialized version of the above function
// that exploits the fact that the column indices for the flux and vector source terms
// are closely related, hence we need not do all sorting operations twice.
std::pair<std::shared_ptr<CompressedDataStorage<double>>,
          std::shared_ptr<CompressedDataStorage<double>>>
create_flux_vector_source_matrix(const std::vector<int>& row_indices,
                                 const std::vector<std::vector<int>>& col_indices,
                                 const std::vector<std::vector<double>>& data_flux,
                                 const std::vector<std::vector<double>>& data_vector_source,
                                 const int num_rows, const int num_cols,
                                 const int tot_num_transmissibilities)
{
    constexpr int SPATIAL_DIM = 3;
    const auto sorting = compute_row_sorting(row_indices, col_indices, num_rows);
    const auto& sorted_row_indices = sorting.sorted_row_indices;
    const auto& num_row_occurrences = sorting.num_row_occurrences;
    const auto& col_index_sizes = sorting.col_index_sizes;

    std::vector<int> row_ptr_flux, row_ptr_vector_source;
    row_ptr_flux.reserve(num_rows + 1);
    row_ptr_flux.push_back(0);
    row_ptr_vector_source.reserve(num_rows + 1);
    row_ptr_vector_source.push_back(0);
    std::vector<int> col_idx_flux, col_idx_vector_source;
    col_idx_flux.reserve(tot_num_transmissibilities);
    col_idx_vector_source.reserve(SPATIAL_DIM * tot_num_transmissibilities);
    std::vector<double> values_flux, values_vector_source;
    values_flux.reserve(tot_num_transmissibilities);
    values_vector_source.reserve(SPATIAL_DIM * tot_num_transmissibilities);

    // A helper structure to store iterated elements in the Array of Structures fashion.
    struct RowEntry
    {
        int col;
        double flux;
        double vector_source[SPATIAL_DIM];
    };
    // Possibly repeated column indices, flux and vector source values in a single row.
    std::vector<RowEntry> row_entries;

    // Starting position in `sorted_row_indices` corresponding to the row we work with in this
    // iteration.
    int current_ind = 0;

    for (int row_ind = 0; row_ind < static_cast<int>(num_row_occurrences.size()); ++row_ind)
    {
        int num_data_this_row = 0;
        for (int i = 0; i < num_row_occurrences[row_ind]; ++i)
        {
            num_data_this_row += col_index_sizes[sorted_row_indices[current_ind + i]];
        }

        row_entries.reserve(num_data_this_row);

        for (int i = 0; i < num_row_occurrences[row_ind]; ++i)
        {
            int loc_sorted_index = sorted_row_indices[current_ind + i];
            const auto& loc_col_indices_flux = col_indices[loc_sorted_index];
            const auto& loc_data_flux = data_flux[loc_sorted_index];
            const auto& loc_data_vector_source = data_vector_source[loc_sorted_index];

            if ((loc_col_indices_flux.size() != loc_data_flux.size()) ||
                (loc_data_flux.size() != (loc_data_vector_source.size() / SPATIAL_DIM)))
            {
                throw std::logic_error("The sizes of the passed vectors do not match.");
            }

            for (int i{0}; i < static_cast<int>(loc_data_flux.size()); ++i)
            {
                row_entries.emplace_back();
                RowEntry& re = row_entries.back();
                re.col = loc_col_indices_flux[i];
                re.flux = loc_data_flux[i];
                std::copy_n(&loc_data_vector_source[i * SPATIAL_DIM], SPATIAL_DIM,
                            re.vector_source);
            }
        }

        if (row_entries.size() == 0)
        {
            // No entries for this row, just copy the previous row pointer.
            row_ptr_flux.push_back(col_idx_flux.size());
            row_ptr_vector_source.push_back(col_idx_vector_source.size());
            continue;
        }

        // Now we need to sort the column indices and data values according to column
        std::sort(row_entries.begin(), row_entries.end(),
                  [](const auto& a, const auto& b) { return a.col < b.col; });

        int prev_col = row_entries[0].col;
        double accum_data_flux = 0.0;
        double accum_data_vector_source[SPATIAL_DIM] = {0.0, 0.0, 0.0};

        for (const auto& row_entry : row_entries)
        {
            if (row_entry.col == prev_col)
            {
                accum_data_flux += row_entry.flux;
                for (int k = 0; k < SPATIAL_DIM; ++k)
                {
                    accum_data_vector_source[k] += row_entry.vector_source[k];
                }
            }
            else
            {
                col_idx_flux.push_back(prev_col);
                values_flux.push_back(accum_data_flux);
                for (int k = 0; k < SPATIAL_DIM; ++k)
                {
                    col_idx_vector_source.push_back(SPATIAL_DIM * prev_col + k);
                    values_vector_source.push_back(accum_data_vector_source[k]);
                }

                prev_col = row_entry.col;
                accum_data_flux = row_entry.flux;
                for (int k = 0; k < SPATIAL_DIM; ++k)
                {
                    accum_data_vector_source[k] = row_entry.vector_source[k];
                }
            }
        }

        // Add the last accumulated value
        col_idx_flux.push_back(prev_col);
        values_flux.push_back(accum_data_flux);
        for (int k = 0; k < SPATIAL_DIM; ++k)
        {
            col_idx_vector_source.push_back(SPATIAL_DIM * prev_col + k);
            values_vector_source.push_back(accum_data_vector_source[k]);
        }

        row_ptr_flux.push_back(col_idx_flux.size());
        row_ptr_vector_source.push_back(col_idx_vector_source.size());

        row_entries.clear();
        current_ind += num_row_occurrences[row_ind];
    }

    // Create the compressed data storage for the flux.
    // EK note to self: The cost of the matrix construction is negligible here.
    auto flux_matrix = std::make_shared<CompressedDataStorage<double>>(
        num_rows, num_cols, std::move(row_ptr_flux), std::move(col_idx_flux),
        std::move(values_flux));
    auto vector_source_matrix = std::make_shared<CompressedDataStorage<double>>(
        num_rows, SPATIAL_DIM * num_cols, std::move(row_ptr_vector_source),
        std::move(col_idx_vector_source), std::move(values_vector_source));

    return {flux_matrix, vector_source_matrix};
}

struct BoundaryFaceClassification
{
    std::vector<std::optional<BoundaryCondition>> types;
    std::vector<std::pair<int, int>> boundary_face_map;
};

// Read-only inputs from the discretisation context, bundled to reduce parameter counts.
struct DiscrContext
{
    const Grid& grid;
    const SecondOrderTensor& tensor;
    const std::vector<int>& num_nodes_of_face;
};

// Local Eigen matrices for one interaction region, bundled to reduce parameter counts.
LocalBalanceMatrices make_local_balance_matrices(int num_faces, int num_cells)
{
    constexpr int SPATIAL_DIM = 3;
    LocalBalanceMatrices m;
    m.balance_cells = Eigen::MatrixXd::Zero(num_faces, num_cells);
    m.balance_faces = Eigen::MatrixXd::Zero(num_faces, num_faces);
    m.flux_cells = Eigen::MatrixXd::Zero(num_faces, num_cells);
    m.flux_faces = Eigen::MatrixXd::Zero(num_faces, num_faces);
    m.nK_matrix = Eigen::MatrixXd::Zero(num_faces, SPATIAL_DIM * num_cells);
    m.nK_one_sided = Eigen::MatrixXd::Zero(num_faces, SPATIAL_DIM * num_cells);
    return m;
}

BoundaryFaceClassification classify_boundary_faces(
    const InteractionRegion& region,
    const std::unordered_map<int, BoundaryCondition>& bc_map)
{
    BoundaryFaceClassification result;
    result.types.resize(region.faces().size(), std::nullopt);
    for (const auto& pair : region.faces())
    {
        auto it = bc_map.find(pair.first);
        if (it != bc_map.end())
        {
            BoundaryCondition bc = it->second;
            if (bc == BoundaryCondition::Dirichlet || bc == BoundaryCondition::Neumann)
            {
                result.boundary_face_map.push_back({pair.second, pair.first});
                result.types.at(pair.second) = bc;
            }
            else if (bc == BoundaryCondition::Robin)
            {
                throw std::logic_error("Robin boundary condition not implemented");
            }
            else
            {
                throw std::logic_error("Unknown boundary condition");
            }
        }
    }
    return result;
}

std::vector<std::array<double, 3>> compute_continuity_points_for_cell(
    int loc_cell_ind,
    const InteractionRegion& region,
    const InteractionRegionGeometry& geom,
    const std::array<double, 3>& node_coord,
    const std::vector<std::optional<BoundaryCondition>>& loc_boundary_faces_type,
    bool is_simplex,
    int dim)
{
    constexpr int SPATIAL_DIM = 3;
    const int glob_cell_ind = region.cells()[loc_cell_ind];
    std::vector<std::array<double, 3>> continuity_points(dim + 1,
                                                         std::array<double, 3>{0.0, 0.0, 0.0});
    continuity_points[0] = geom.cell_centers[loc_cell_ind];
    int face_counter = 1;
    for (const int glob_face_ind : region.faces_of_cells().at(glob_cell_ind))
    {
        const int loc_face_index = region.faces().at(glob_face_ind);
        bool in_dir = false;
        bool in_neu = false;
        if (const auto bc = loc_boundary_faces_type[loc_face_index]; bc.has_value())
        {
            in_dir = *bc == BoundaryCondition::Dirichlet;
            in_neu = *bc == BoundaryCondition::Neumann;
        }
        if (is_simplex && (!in_dir) && (!in_neu))
        {
            for (int i = 0; i < SPATIAL_DIM; ++i)
            {
                continuity_points[face_counter][i] =
                    2.0 / 3 * geom.face_centers[loc_face_index][i] +
                    (1.0 / 3) * node_coord[i];
            }
        }
        else
        {
            for (int k{0}; k < SPATIAL_DIM; ++k)
            {
                continuity_points[face_counter][k] = geom.face_centers[loc_face_index][k];
            }
        }
        ++face_counter;
    }
    return continuity_points;
}

void fill_cell_contributions(
    int loc_cell_ind,
    const InteractionRegion& region,
    const DiscrContext& ctx,
    const InteractionRegionGeometry& geom,
    const std::vector<std::optional<BoundaryCondition>>& loc_boundary_faces_type,
    const std::vector<std::array<double, 3>>& basis_functions,
    LocalBalanceMatrices& matrices)
{
    constexpr int SPATIAL_DIM = 3;
    const int glob_cell_ind = region.cells()[loc_cell_ind];

    // Build face index vectors for this cell (declared within to allow OpenMP later).
    std::vector<int> loc_faces_of_cell;
    std::vector<int> glob_faces_of_cell;
    for (const int glob_face_ind : region.faces_of_cells().at(glob_cell_ind))
    {
        glob_faces_of_cell.push_back(glob_face_ind);
        loc_faces_of_cell.push_back(region.faces().at(glob_face_ind));
    }
    const int dim = static_cast<int>(loc_faces_of_cell.size());

    for (int outer_face_counter{0}; outer_face_counter < dim; ++outer_face_counter)
    {
        const int glob_face_ind = glob_faces_of_cell[outer_face_counter];
        const int loc_face_index = loc_faces_of_cell[outer_face_counter];

        std::array<double, 3> flux_expr =
            nK(geom.face_normals[loc_face_index], ctx.tensor, glob_cell_ind,
               ctx.num_nodes_of_face[glob_face_ind]);
        const int sign = ctx.grid.sign_of_face_cell(glob_face_ind, glob_cell_ind);

        std::vector<double> flux_vals = nKgrad(flux_expr, basis_functions);

        std::vector<double> dirichlet_vals;

        bool is_boundary_face = false;
        BoundaryCondition bc{};
        if (const auto optional_bc = loc_boundary_faces_type[loc_face_index];
            optional_bc.has_value())
        {
            is_boundary_face = true;
            bc = *optional_bc;
        }

        if (is_boundary_face && (bc == BoundaryCondition::Dirichlet))
        {
            dirichlet_vals = p_diff(geom.face_centers[loc_face_index],
                                    geom.cell_centers[loc_cell_ind], basis_functions);
            matrices.balance_cells(loc_face_index, loc_cell_ind) = -dirichlet_vals[0] - 1.0;
        }
        else
        {
            matrices.balance_cells(loc_face_index, loc_cell_ind) = -sign * flux_vals[0];
            for (int i = 0; i < SPATIAL_DIM; ++i)
            {
                const int col = i + SPATIAL_DIM * loc_cell_ind;
                matrices.nK_matrix(loc_face_index, col) = sign * flux_expr[i];
            }
        }

        for (int i = 1; i < dim + 1; ++i)
        {
            const int loc_face_index_secondary = loc_faces_of_cell[i - 1];
            if (is_boundary_face && (bc == BoundaryCondition::Dirichlet))
            {
                matrices.balance_faces(loc_face_index, loc_face_index_secondary) +=
                    dirichlet_vals[i];
            }
            else
            {
                matrices.balance_faces(loc_face_index, loc_face_index_secondary) +=
                    sign * flux_vals[i];
            }
        }

        if (glob_cell_ind == region.main_cell_of_faces().at(loc_face_index))
        {
            matrices.flux_cells(loc_face_index, loc_cell_ind) = flux_vals[0];
            for (int i = 1; i < dim + 1; ++i)
            {
                const int glob_face_index_secondary =
                    region.faces_of_cells().at(glob_cell_ind)[i - 1];
                const int loc_face_index_secondary =
                    region.faces().at(glob_face_index_secondary);
                matrices.flux_faces(loc_face_index, loc_face_index_secondary) = flux_vals[i];
            }
            // nK_one_sided is set for all faces, including Dirichlet boundary faces.
            // This matches the reference PorePy implementation.
            for (int k = 0; k < SPATIAL_DIM; ++k)
            {
                matrices.nK_one_sided(loc_face_index, k + loc_cell_ind * SPATIAL_DIM) =
                    -flux_expr[k];
            }
        }

        if (is_boundary_face)
        {
            matrices.basis_map[glob_cell_ind] = basis_functions;
        }
    }
}

struct LocalFluxMatrices
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> balance_faces_inv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bound_flux;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flux;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vector_source_cell;
};

LocalFluxMatrices compute_local_flux(
    const LocalBalanceMatrices& matrices,
    const std::vector<std::optional<BoundaryCondition>>& loc_boundary_faces_type)
{
    LocalFluxMatrices result;

    result.balance_faces_inv = matrices.balance_faces.inverse();
    result.bound_flux.noalias() = matrices.flux_faces * result.balance_faces_inv;

    bool has_dirichlet = false;
    for (const auto& bc : loc_boundary_faces_type)
    {
        if (bc.has_value() && *bc == BoundaryCondition::Dirichlet)
        {
            has_dirichlet = true;
            break;
        }
    }

    if (!has_dirichlet)
    {
        result.flux.noalias() = result.bound_flux * matrices.balance_cells + matrices.flux_cells;
    }
    else
    {
        const int num_faces = static_cast<int>(loc_boundary_faces_type.size());
        Eigen::VectorXd mask = Eigen::VectorXd::Ones(num_faces);
        for (int face{0}; face < num_faces; ++face)
        {
            const auto bc = loc_boundary_faces_type[face];
            if (bc.has_value() && *bc == BoundaryCondition::Dirichlet)
            {
                mask(face) = 0.0;
            }
        }
        result.flux.noalias() =
            result.bound_flux * mask.asDiagonal() * matrices.balance_cells + matrices.flux_cells;
    }

    result.vector_source_cell.noalias() =
        result.bound_flux * matrices.nK_matrix + matrices.nK_one_sided;
    return result;
}

FluxStencilData init_flux_stencil(int num_faces, int nodes_per_face, int dim)
{
    const int capacity = num_faces * nodes_per_face + 1;
    FluxStencilData s;
    s.row_idx.reserve(capacity);
    s.col_idx.reserve(capacity);
    s.flux_values.reserve(capacity);
    s.vs_values.reserve(num_faces * dim * nodes_per_face + 1);
    return s;
}

BoundaryStencilData init_boundary_stencil(int num_faces, int num_bound_faces,
                                          int nodes_per_face, int dim)
{
    BoundaryStencilData s;
    s.bound_flux.row_idx.reserve(num_faces * nodes_per_face);
    s.bound_flux.col_idx.reserve(num_bound_faces * nodes_per_face);
    s.bound_flux.values.reserve(num_bound_faces * nodes_per_face);

    s.pressure_cell.values.reserve(num_bound_faces * nodes_per_face);
    s.pressure_cell.row_idx.reserve(num_faces * nodes_per_face);
    s.pressure_cell.col_idx.reserve(num_bound_faces * nodes_per_face);

    s.pressure_face.values.reserve(num_bound_faces * nodes_per_face);
    s.pressure_face.row_idx.reserve(num_faces * nodes_per_face);
    s.pressure_face.col_idx.reserve(num_bound_faces * nodes_per_face);

    s.vector_source.values.reserve(num_bound_faces * dim * nodes_per_face + 1);
    s.vector_source.row_idx.reserve(num_faces * dim * nodes_per_face + 1);
    s.vector_source.col_idx.reserve(num_bound_faces * dim * nodes_per_face + 1);
    return s;
}

// Accumulate boundary discretization data for a single interaction region into
// the output accumulators. Does nothing if the interaction region has no boundary faces.
void accumulate_boundary_data(
    const InteractionRegion& interaction_region,
    const LocalFluxMatrices& local_flux,
    const BoundaryFaceClassification& bc_class,
    const InteractionRegionGeometry& geom,
    const LocalBalanceMatrices& matrices,
    const std::vector<int>& num_nodes_of_face,
    BoundaryStencilData& out)
{
    const auto& loc_boundary_face_map = bc_class.boundary_face_map;
    if (loc_boundary_face_map.empty())
    {
        return;
    }

    constexpr int SPATIAL_DIM = 3;

    const auto& loc_boundary_faces_type = bc_class.types;
    const auto& balance_faces_inv = local_flux.balance_faces_inv;
    const auto& bound_flux = local_flux.bound_flux;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        bound_vector_source_matrix;
    bound_vector_source_matrix.noalias() = balance_faces_inv * matrices.nK_matrix;

    // For each face in the interaction region (internal + boundary), build the
    // bound_flux discretization entries.
    for (const auto& face_pair : interaction_region.faces())
    {
        out.bound_flux.row_idx.emplace_back(face_pair.first);
        out.bound_flux.col_idx.emplace_back();
        out.bound_flux.values.emplace_back();
        auto& bf_indices = out.bound_flux.col_idx.back();
        auto& bf_val = out.bound_flux.values.back();

        for (const auto& loc_face_pair : loc_boundary_face_map)
        {
            if (const auto bc = loc_boundary_faces_type[loc_face_pair.first];
                bc.has_value() && *bc == BoundaryCondition::Neumann)
            {
                // Scale by 1/num_nodes: the BC is given as total flux over the face.
                bf_val.push_back(bound_flux(face_pair.second, loc_face_pair.first) /
                                 num_nodes_of_face[loc_face_pair.first]);
            }
            else
            {
                bf_val.push_back(bound_flux(face_pair.second, loc_face_pair.first));
            }
            bf_indices.push_back(loc_face_pair.second);
        }
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        face_pressure_from_cells;
    // Mapping from cell center pressure to face pressures.
    face_pressure_from_cells.noalias() = balance_faces_inv * matrices.balance_cells;

    // Collect global face indices once (reused per boundary face below).
    std::vector<int> glob_indices_iareg_faces;
    glob_indices_iareg_faces.reserve(interaction_region.faces().size());
    for (const auto& face_pair : interaction_region.faces())
    {
        glob_indices_iareg_faces.push_back(face_pair.first);
    }

    // For each boundary face, build pressure-reconstruction and vector-source entries.
    for (const auto& loc_face_pair : loc_boundary_face_map)
    {
        const double inv_num_nodes_of_face = 1.0 / num_nodes_of_face[loc_face_pair.first];

        if (const auto bc = loc_boundary_faces_type[loc_face_pair.first];
            bc.has_value() && *bc == BoundaryCondition::Dirichlet)
        {
            // For a Dirichlet boundary face, only a unit face contribution is needed.
            out.pressure_face.values.emplace_back(
                std::initializer_list<double>{1.0 * inv_num_nodes_of_face});
            out.pressure_face.row_idx.push_back(loc_face_pair.second);
            out.pressure_face.col_idx.emplace_back(
                std::initializer_list<int>{loc_face_pair.second});
            continue;
        }

        // Find the cell adjacent to the boundary face and its local index.
        const int glob_cell_ind =
            interaction_region.main_cell_of_faces().at(loc_face_pair.first);
        const int loc_cell_ind =
            std::find(interaction_region.cells().begin(), interaction_region.cells().end(),
                      glob_cell_ind) -
            interaction_region.cells().begin();

        // Pressure difference between the face center and its adjacent cell center.
        const std::vector<double> pressure_diff =
            p_diff(geom.face_centers[loc_face_pair.first], geom.cell_centers[loc_cell_ind],
                   matrices.basis_map.at(glob_cell_ind));

        std::vector<double> cell_contribution(interaction_region.cells().size(), 0.0);
        std::vector<double> vector_source_cell_contribution(
            interaction_region.cells().size() * SPATIAL_DIM, 0.0);

        // The cell itself contributes a unit value (offset) plus the gradient correction.
        cell_contribution[loc_cell_ind] = (1.0 + pressure_diff[0]) * inv_num_nodes_of_face;

        std::vector<double> face_contribution(interaction_region.faces().size(), 0.0);

        // Loop over the faces of the cell that also belong to the interaction region.
        // Start at 1, since the first basis function is the cell center pressure.
        int basis_vector_face_counter = 1;
        for (const int face_ind : interaction_region.faces_of_cells().at(glob_cell_ind))
        {
            const int face_local_index = interaction_region.faces().at(face_ind);

            std::vector<double> row_from_cells(
                face_pressure_from_cells.row(face_local_index).data(),
                face_pressure_from_cells.row(face_local_index).data() +
                    face_pressure_from_cells.row(face_local_index).size());

            std::vector<double> row_from_cells_vector_source(
                bound_vector_source_matrix.row(face_local_index).data(),
                bound_vector_source_matrix.row(face_local_index).data() +
                    bound_vector_source_matrix.row(face_local_index).size());

            for (int loc_cell_ind{0};
                 loc_cell_ind < static_cast<int>(interaction_region.cells().size());
                 ++loc_cell_ind)
            {
                cell_contribution[loc_cell_ind] +=
                    row_from_cells[loc_cell_ind] *
                    pressure_diff[basis_vector_face_counter] * inv_num_nodes_of_face;
                for (int k = 0; k < SPATIAL_DIM; ++k)
                {
                    const int col = loc_cell_ind * SPATIAL_DIM + k;
                    vector_source_cell_contribution[col] +=
                        row_from_cells_vector_source[col] *
                        pressure_diff[basis_vector_face_counter] * inv_num_nodes_of_face;
                }
            }

            std::vector<double> row_from_faces(
                balance_faces_inv.row(face_local_index).data(),
                balance_faces_inv.row(face_local_index).data() +
                    balance_faces_inv.row(face_local_index).size());

            for (const auto& face_pair : interaction_region.faces())
            {
                const int loc_face_ind = face_pair.second;
                if (const auto bc = loc_boundary_faces_type[loc_face_ind]; bc.has_value())
                {
                    double contribution_from_face =
                        row_from_faces[loc_face_ind] *
                        pressure_diff[basis_vector_face_counter] * inv_num_nodes_of_face;

                    if (*bc == BoundaryCondition::Neumann)
                    {
                        contribution_from_face *= inv_num_nodes_of_face;
                    }

                    face_contribution[loc_face_ind] += contribution_from_face;
                }
            }
            ++basis_vector_face_counter;
        }

        out.pressure_cell.values.push_back(std::move(cell_contribution));
        out.pressure_cell.row_idx.push_back(loc_face_pair.second);
        out.pressure_cell.col_idx.push_back(interaction_region.cells());
        out.pressure_face.values.push_back(std::move(face_contribution));
        out.pressure_face.row_idx.push_back(loc_face_pair.second);
        out.pressure_face.col_idx.push_back(glob_indices_iareg_faces);

        out.vector_source.values.push_back(std::move(vector_source_cell_contribution));
        out.vector_source.row_idx.push_back(loc_face_pair.second);

        std::vector<int> cell_ind_vector_source;
        cell_ind_vector_source.reserve(interaction_region.cells().size() * SPATIAL_DIM);
        for (const auto& cell_ind : interaction_region.cells())
        {
            for (int k = 0; k < SPATIAL_DIM; ++k)
            {
                cell_ind_vector_source.push_back(cell_ind * SPATIAL_DIM + k);
            }
        }
        out.vector_source.col_idx.push_back(std::move(cell_ind_vector_source));
    }
}

// Overload that accepts FluxStencilData directly.
std::pair<std::shared_ptr<CompressedDataStorage<double>>,
          std::shared_ptr<CompressedDataStorage<double>>>
create_flux_vector_source_matrix(const FluxStencilData& stencil, const int num_rows,
                                 const int num_cols, const int tot_num_transmissibilities)
{
    return create_flux_vector_source_matrix(stencil.row_idx, stencil.col_idx, stencil.flux_values,
                                            stencil.vs_values, num_rows, num_cols,
                                            tot_num_transmissibilities);
}

}  // namespace mpfa_detail

ScalarDiscretization mpfa(const Grid& grid, const SecondOrderTensor& tensor,
                          const std::unordered_map<int, BoundaryCondition>& bc_map)
{
    using namespace mpfa_detail;

    // MPFA reduces to TPFA in 1D or 0D. We do it explicitly here, and further code
    // assumes a 2D or 3D grid.
    if (grid.dim() < 2) {
        return tpfa(grid, tensor, bc_map);
    }
    constexpr int SPATIAL_DIM = 3;  // Assuming 3D for now, can be generalized later.

    BasisConstructor basis_constructor(grid.dim());

    std::vector<int> num_nodes_of_face = count_nodes_of_faces(grid);
    std::vector<int> num_faces_of_cell = count_faces_of_cells(grid);

    const int num_bound_faces = bc_map.size();
    const int DIM = grid.dim();

    // Data structures for the computed stencils.
    auto flux_stencil = init_flux_stencil(grid.num_faces(), num_nodes_of_face[0], DIM);
    auto bound_out = init_boundary_stencil(grid.num_faces(), num_bound_faces,
                                           num_nodes_of_face[0], DIM);

    int tot_num_transmissibilities = 0;
    const DiscrContext ctx{grid, tensor, num_nodes_of_face};

    // NOTE: The node loop body is embarrassingly parallel — suitable for #pragma omp parallel for
    for (int node_ind{0}; node_ind < grid.num_nodes(); ++node_ind)
    {
        // Get the interaction region for the node.
        InteractionRegion interaction_region(node_ind, 1, grid);

        const int num_faces = interaction_region.faces().size();
        const int num_cells = interaction_region.cells().size();

        auto matrices = make_local_balance_matrices(num_faces, num_cells);

        const auto geom = compute_interaction_region_geometry(interaction_region, grid);

        // If all cells have grid.dim() + 1 faces, this is a simplex. Use a boolean to
        // indicate whether this is a simplex or not.
        const bool is_simplex = check_if_simplex(num_faces_of_cell, DIM);

        // Classify boundary faces within this interaction region.
        auto bc_classification = classify_boundary_faces(interaction_region, bc_map);
        auto& loc_boundary_faces_type = bc_classification.types;
        auto& loc_boundary_face_map = bc_classification.boundary_face_map;

        const std::array<double, 3> node_coord_arr = to_array3(grid.nodes()[node_ind]);

        // For each cell: compute continuity points, basis functions, and fill contributions.
        for (int loc_cell_ind{0}; loc_cell_ind < num_cells; ++loc_cell_ind)
        {
            const auto continuity_pts = compute_continuity_points_for_cell(
                loc_cell_ind, interaction_region, geom,
                node_coord_arr, loc_boundary_faces_type, is_simplex, DIM);

            const auto basis_fns =
                basis_constructor.compute_basis_functions(continuity_pts);

            fill_cell_contributions(
                loc_cell_ind, interaction_region, ctx, geom,
                loc_boundary_faces_type, basis_fns, matrices);
        }  // End iteration of cells of the interaction region.

        // Compute the local flux matrices (inversion + mask logic).
        const auto local_flux = compute_local_flux(matrices, loc_boundary_faces_type);
        const auto& flux = local_flux.flux;
        const auto& vector_source_cell = local_flux.vector_source_cell;

        // Store the computed flux in the flux_matrix_values, row_idx, and col_idx.
        size_t vs_cols = vector_source_cell.cols();
        size_t cols = flux.cols();
        for (const auto face_inds : interaction_region.faces())
        {
            long row_id = face_inds.second;

            // Constructing the vector IN PLACE at the end of the list. This avoids
            // creating a temporary vector and then moving/copying it.
            flux_stencil.flux_values.emplace_back();
            auto& new_flux_row = flux_stencil.flux_values.back();
            new_flux_row.resize(cols);

            // Copying raw data. We use RowMajor format, so data is contiguous.
            std::memcpy(new_flux_row.data(), flux.row(row_id).data(), cols * sizeof(double));

            flux_stencil.row_idx.push_back(face_inds.first);

            // Same for Vector Source.
            flux_stencil.vs_values.emplace_back();
            auto& new_vs_row = flux_stencil.vs_values.back();
            new_vs_row.resize(vs_cols);
            std::memcpy(new_vs_row.data(), vector_source_cell.row(row_id).data(),
                        vs_cols * sizeof(double));
        }
        // Store column indices for the flux matrix.
        for (int i = 0; i < num_faces; ++i)
        {
            flux_stencil.col_idx.emplace_back(interaction_region.cells());
        }

        tot_num_transmissibilities += num_faces * num_cells;

        accumulate_boundary_data(
            interaction_region, local_flux, bc_classification, geom, matrices,
            num_nodes_of_face, bound_out);
    }  // End iteration of nodes in the grid.

    // Gather the computed data into CSR matrices and further into the discretization
    // structure.
    ScalarDiscretization discretization;

    // Use a tailored method for creating CSR matrices for the flux and vector source
    // matrices, since they have similar sparsity pattern, and the construction of these
    // take considerable time.
    auto flux_and_vs = create_flux_vector_source_matrix(
        flux_stencil, grid.num_faces(), grid.num_cells(), tot_num_transmissibilities);
    discretization.flux = flux_and_vs.first;
    discretization.vector_source = flux_and_vs.second;

    // For the remaining matrices we use a more general method. There could be some
    // savings to be gained here as well, but in the bigger picture, the time spent
    // here is not that important.
    discretization.bound_flux = create_csr_matrix(
        bound_out.bound_flux.row_idx, bound_out.bound_flux.col_idx, bound_out.bound_flux.values,
        grid.num_faces(), grid.num_faces(), bound_out.bound_flux.row_idx.size());

    discretization.bound_pressure_cell = create_csr_matrix(
        bound_out.pressure_cell.row_idx, bound_out.pressure_cell.col_idx,
        bound_out.pressure_cell.values, grid.num_faces(), grid.num_cells(),
        bound_out.pressure_cell.row_idx.size());

    discretization.bound_pressure_face = create_csr_matrix(
        bound_out.pressure_face.row_idx, bound_out.pressure_face.col_idx,
        bound_out.pressure_face.values, grid.num_faces(), grid.num_faces(),
        bound_out.pressure_face.row_idx.size());

    discretization.bound_pressure_vector_source = create_csr_matrix(
        bound_out.vector_source.row_idx, bound_out.vector_source.col_idx,
        bound_out.vector_source.values, grid.num_faces(), SPATIAL_DIM * grid.num_cells(),
        bound_out.vector_source.row_idx.size());

    return discretization;
}