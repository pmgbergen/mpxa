#include <Eigen/Dense>
#include <array>
#include <map>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "../include/discr.h"
#include "../include/multipoint_common.h"

using Eigen::MatrixXd;

namespace
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
void cell_centers_of_interaction_region(const InteractionRegion& interaction_region,
                                        const Grid& grid,
                                        std::vector<std::array<double, 3>>& centers)
{
    centers.clear();
    for (const int cell_ind : interaction_region.cells())
    {
        const std::vector<double>& center_vec = grid.cell_center(cell_ind);
        std::array<double, 3> center_arr = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < std::min(center_vec.size(), center_arr.size()); ++i)
        {
            center_arr[i] = center_vec[i];
        }
        centers.push_back(center_arr);
    }
}

// Helper function to get face center coordinates for all faces in an interaction region
void face_centers_of_interaction_region(const InteractionRegion& interaction_region,
                                        const Grid& grid,
                                        std::vector<std::array<double, 3>>& centers)
{
    centers.clear();
    for (const auto& pair : interaction_region.faces())
    {
        const std::vector<double>& face_center = grid.face_center(pair.first);
        std::array<double, 3> center_arr = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < std::min(face_center.size(), center_arr.size()); ++i)
        {
            center_arr[i] = face_center[i];
        }
        centers.push_back(center_arr);
    }
}

// Helper function to get face normals for all faces in an interaction region
void face_normals_of_interaction_region(const InteractionRegion& interaction_region,
                                        const Grid& grid,
                                        std::vector<std::array<double, 3>>& normals)
{
    normals.clear();
    for (const auto& pair : interaction_region.faces())
    {
        const std::vector<double>& face_normal = grid.face_normal(pair.first);
        std::array<double, 3> normal_arr = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < std::min(face_normal.size(), normal_arr.size()); ++i)
        {
            normal_arr[i] = face_normal[i];
        }
        normals.push_back(normal_arr);
    }
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

    for (int i{0}; i < col_idx.size(); ++i)
    {
        ++num_nodes_of_face[col_idx[i]];
    }

    return num_nodes_of_face;
}

std::vector<int> count_faces_of_cells(const Grid& grid)
{
    // Count the number of faces for each cell in the grid.
    std::vector<int> num_faces_of_cell(grid.num_cells(), 0);
    const auto& cell_faces = grid.cell_faces();
    const auto& col_idx = cell_faces.col_idx();
    for (int i{0}; i < col_idx.size(); ++i)
    {
        num_faces_of_cell[col_idx[i]]++;
    }
    return num_faces_of_cell;
}

struct PairHash
{
    std::size_t operator()(const std::pair<int, int>& p) const noexcept
    {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// Helper function to create a compressed sparse row (CSR) matrix from vectors.
std::shared_ptr<CompressedDataStorage<double>> create_csr_matrix(
    const std::vector<int>& row_indices, const std::vector<std::vector<int>>& col_indices,
    const std::vector<std::vector<double>>& data_values, const int num_rows, const int num_cols,
    const int tot_num_transmissibilities)
{
    std::vector<int> sorted_row_indices(row_indices.size());
    std::iota(sorted_row_indices.begin(), sorted_row_indices.end(), 0);
    std::sort(sorted_row_indices.begin(), sorted_row_indices.end(),
              [&row_indices](int i1, int i2) { return row_indices[i1] < row_indices[i2]; });

    std::vector<int> row_ptr;
    row_ptr.reserve(num_rows + 1);
    row_ptr.push_back(0);
    std::vector<int> col_idx;
    col_idx.reserve(tot_num_transmissibilities);
    std::vector<double> values;
    values.reserve(tot_num_transmissibilities);

    std::vector<int> num_row_occurrences(num_rows, 0);
    for (const int row_ind : row_indices)
    {
        ++num_row_occurrences[row_ind];
    }
    std::vector<int> col_index_sizes;
    col_index_sizes.reserve(col_indices.size());
    for (const auto& vec : col_indices)
    {
        col_index_sizes.push_back(vec.size());
    }

    // Local storage for the sorted column indices and data values for each row. Will be
    // erased for each iteration.
    std::vector<int> sorted_col_indices;
    std::vector<double> sorted_data_values;
    std::vector<int> this_row_col_indices;
    std::vector<double> this_row_data;

    int current_ind = 0;
    for (int row_ind = 0; row_ind < num_row_occurrences.size(); ++row_ind)
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
    auto matrix = std::make_shared<CompressedDataStorage<double>>(num_rows, num_cols, row_ptr,
                                                                  col_idx, values);
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
    // Get the sorted indices for the row indices. This is common for the flux and
    // vector source data.
    std::vector<int> sorted_row_indices(row_indices.size());
    std::iota(sorted_row_indices.begin(), sorted_row_indices.end(), 0);
    std::sort(sorted_row_indices.begin(), sorted_row_indices.end(),
              [&row_indices](int i1, int i2) { return row_indices[i1] < row_indices[i2]; });

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

    // Count the number of occurrences of each row index.
    std::vector<int> num_row_occurrences(num_rows, 0);
    for (const int row_ind : row_indices)
    {
        ++num_row_occurrences[row_ind];
    }
    std::vector<int> col_index_sizes;
    col_index_sizes.reserve(col_indices.size());
    for (const auto& vec : col_indices)
    {
        col_index_sizes.push_back(vec.size());
    }

    std::vector<int> sorted_col_indices_flux, sorted_col_indices_vector_source;
    std::vector<double> sorted_data_values_flux, sorted_data_values_vector_source;
    std::vector<int> this_row_col_indices_flux, this_row_col_indices_vector_source;
    std::vector<double> this_row_data_flux, this_row_data_vector_source;
    std::vector<int> loc_sorted_indices, sorting_col_indices;

    int current_ind = 0;
    for (int row_ind = 0; row_ind < num_row_occurrences.size(); ++row_ind)
    {
        if (num_row_occurrences[row_ind] == 0)
        {
            // No entries for this row, just copy the previous row pointer.
            row_ptr_flux.push_back(row_ptr_flux.back());
            row_ptr_vector_source.push_back(row_ptr_vector_source.back());
            continue;
        }
        loc_sorted_indices.reserve(num_row_occurrences[row_ind]);
        for (int i = 0; i < num_row_occurrences[row_ind]; ++i)
        {
            loc_sorted_indices.push_back(sorted_row_indices[current_ind + i]);
        }
        current_ind += num_row_occurrences[row_ind];

        int num_data_this_row = 0;
        for (int i = 0; i < num_row_occurrences[row_ind]; ++i)
        {
            //
            num_data_this_row += col_index_sizes[loc_sorted_indices[i]];
        }

        this_row_col_indices_flux.reserve(num_data_this_row);
        this_row_data_flux.reserve(num_data_this_row);
        this_row_col_indices_vector_source.reserve(SPATIAL_DIM * num_data_this_row);
        this_row_data_vector_source.reserve(SPATIAL_DIM * num_data_this_row);

        for (int i = 0; i < num_row_occurrences[row_ind]; ++i)
        {
            const auto& loc_col_indices_flux = col_indices[loc_sorted_indices[i]];
            const auto& loc_data_flux = data_flux[loc_sorted_indices[i]];
            const auto& loc_data_vector_source = data_vector_source[loc_sorted_indices[i]];
            this_row_col_indices_flux.insert(this_row_col_indices_flux.end(),
                                             loc_col_indices_flux.begin(),
                                             loc_col_indices_flux.end());
            this_row_data_flux.insert(this_row_data_flux.end(), loc_data_flux.begin(),
                                      loc_data_flux.end());
            // For the vector source, we need to add SPATIAL_DIM entries for each
            // column index in the flux data.
            for (size_t j = 0; j < loc_col_indices_flux.size(); ++j)
            {
                for (int k = 0; k < SPATIAL_DIM; ++k)
                {
                    this_row_col_indices_vector_source.push_back(
                        SPATIAL_DIM * loc_col_indices_flux[j] + k);
                }
            }
            this_row_data_vector_source.insert(this_row_data_vector_source.end(),
                                               loc_data_vector_source.begin(),
                                               loc_data_vector_source.end());
        }

        if (this_row_col_indices_flux.size() == 0)
        {
            // No entries for this row, just copy the previous row pointer.
            row_ptr_flux.push_back(col_idx_flux.size());
            row_ptr_vector_source.push_back(col_idx_vector_source.size());
            continue;
        }

        // Now we need to sort the column indices and data values according to column
        sorting_col_indices.reserve(this_row_col_indices_flux.size());
        for (auto i = 0; i < this_row_col_indices_flux.size(); ++i) {
            sorting_col_indices.push_back(i);
        }
        std::sort(sorting_col_indices.begin(), sorting_col_indices.end(),
                  [&this_row_col_indices_flux](int a, int b)
                  { return this_row_col_indices_flux[a] < this_row_col_indices_flux[b]; });

        // Create the sorted column indices and data values.
        sorted_col_indices_flux.reserve(this_row_col_indices_flux.size());
        sorted_data_values_flux.reserve(this_row_data_flux.size());
        sorted_col_indices_vector_source.reserve(this_row_col_indices_vector_source.size());
        sorted_data_values_vector_source.reserve(this_row_data_vector_source.size());

        int prev_col = this_row_col_indices_flux[sorting_col_indices[0]];
        double accum_data_flux = 0.0;
        double accum_data_vector_source[SPATIAL_DIM] = {0.0, 0.0, 0.0};

        for (int i : sorting_col_indices)
        {
            if (this_row_col_indices_flux[i] == prev_col)
            {
                accum_data_flux += this_row_data_flux[i];
                for (int k = 0; k < SPATIAL_DIM; ++k)
                {
                    accum_data_vector_source[k] += this_row_data_vector_source[SPATIAL_DIM * i + k];
                }
            }
            else
            {
                sorted_col_indices_flux.push_back(prev_col);
                sorted_data_values_flux.push_back(accum_data_flux);
                for (int k = 0; k < SPATIAL_DIM; ++k)
                {
                    sorted_col_indices_vector_source.push_back(SPATIAL_DIM * prev_col + k);
                    sorted_data_values_vector_source.push_back(accum_data_vector_source[k]);
                }

                prev_col = this_row_col_indices_flux[i];
                accum_data_flux = this_row_data_flux[i];
                for (int k = 0; k < SPATIAL_DIM; ++k)
                {
                    accum_data_vector_source[k] = this_row_data_vector_source[SPATIAL_DIM * i + k];
                }
            }
        }

        // Add the last accumulated value
        sorted_col_indices_flux.push_back(prev_col);
        sorted_data_values_flux.push_back(accum_data_flux);

        for (int k = 0; k < SPATIAL_DIM; ++k)
        {
            sorted_col_indices_vector_source.push_back(SPATIAL_DIM * prev_col + k);
            sorted_data_values_vector_source.push_back(accum_data_vector_source[k]);
        }

        col_idx_flux.insert(col_idx_flux.end(), sorted_col_indices_flux.begin(),
                            sorted_col_indices_flux.end());
        values_flux.insert(values_flux.end(), sorted_data_values_flux.begin(),
                           sorted_data_values_flux.end());
        col_idx_vector_source.insert(col_idx_vector_source.end(),
                                     sorted_col_indices_vector_source.begin(),
                                     sorted_col_indices_vector_source.end());
        values_vector_source.insert(values_vector_source.end(),
                                    sorted_data_values_vector_source.begin(),
                                    sorted_data_values_vector_source.end());
        row_ptr_flux.push_back(col_idx_flux.size());
        row_ptr_vector_source.push_back(col_idx_vector_source.size());

        this_row_col_indices_flux.clear();
        this_row_data_flux.clear();
        sorted_col_indices_flux.clear();
        sorted_data_values_flux.clear();
        this_row_col_indices_vector_source.clear();
        this_row_data_vector_source.clear();
        sorted_col_indices_vector_source.clear();
        sorted_data_values_vector_source.clear();
        loc_sorted_indices.clear();
        sorting_col_indices.clear();
    }

    // Create the compressed data storage for the flux.
    // EK note to self: The cost of the matrix construction is negligible here.
    auto flux_matrix = std::make_shared<CompressedDataStorage<double>>(
        num_rows, num_cols, row_ptr_flux, col_idx_flux, values_flux);
    auto vector_source_matrix = std::make_shared<CompressedDataStorage<double>>(
        num_rows, SPATIAL_DIM * num_cols, row_ptr_vector_source, col_idx_vector_source,
        values_vector_source);

    return {flux_matrix, vector_source_matrix};
}

}  // namespace

ScalarDiscretization mpfa(const Grid& grid, const SecondOrderTensor& tensor,
                          const std::unordered_map<int, BoundaryCondition>& bc_map)
{
    constexpr int SPATIAL_DIM = 3;  // Assuming 3D for now, can be generalized later.

    BasisConstructor basis_constructor(grid.dim());
    // Data structures for the discretization process. There will be grid.dim() continuity points
    // (outer vector).
    std::vector<std::array<double, 3>> continuity_points(grid.dim() + 1,
                                                         std::array<double, 3>{0.0, 0.0, 0.0});

    std::vector<std::array<double, 3>> basis_functions(grid.dim() + 1,
                                                       std::array<double, 3>{0.0, 0.0, 0.0});

    std::vector<int> num_nodes_of_face = count_nodes_of_faces(grid);
    std::vector<int> num_faces_of_cell = count_faces_of_cells(grid);

    // YZ: We are not using the things below.
    int avg_num_cells_per_node;
    int avg_num_cell_per_bound_node;
    if (grid.dim() == 2)
    {
        if (num_faces_of_cell[0] == 4)
        {
            avg_num_cells_per_node = 4;  // Rough estimate for structured quadrilateral grids.
            avg_num_cell_per_bound_node = 2;
        }
        else
        {
            avg_num_cells_per_node = 6;  // Rough estimate for 2D grids.
            avg_num_cell_per_bound_node = 3;
        }
    }
    else
    {
        if (num_faces_of_cell[0] == 6)
        {
            avg_num_cells_per_node = 8;  // Rough estimate for structured hexahedral grids.
            avg_num_cell_per_bound_node = 4;
        }
        else
        {
            avg_num_cells_per_node = 14;  // Rough estimate for 3D grids.
            avg_num_cell_per_bound_node = 6;
        }
    }
    const int num_bound_faces = bc_map.size();

    // Data structures for the computed stencils.
    std::vector<int> flux_matrix_row_idx;
    flux_matrix_row_idx.reserve(grid.num_faces() * num_nodes_of_face[0] + 1);
    std::vector<std::vector<int>> flux_matrix_col_idx;
    flux_matrix_col_idx.reserve(grid.num_faces() * num_nodes_of_face[0] + 1);
    std::vector<std::vector<double>> flux_matrix_values;
    flux_matrix_values.reserve(grid.num_faces() * num_nodes_of_face[0] + 1);

    // Data structures for the discretization of boundary conditions.
    std::vector<int> bound_flux_matrix_row_idx;
    bound_flux_matrix_row_idx.reserve(grid.num_faces() * num_nodes_of_face[0]);
    std::vector<std::vector<int>> bound_flux_matrix_col_idx;
    bound_flux_matrix_col_idx.reserve(num_bound_faces * num_nodes_of_face[0]);
    std::vector<std::vector<double>> bound_flux_matrix_values;
    bound_flux_matrix_values.reserve(num_bound_faces * num_nodes_of_face[0]);

    // Data structures for pressure reconstruction on boundary faces. Cell contributions.
    std::vector<std::vector<double>> pressure_reconstruction_cell_values;
    pressure_reconstruction_cell_values.reserve(num_bound_faces * num_nodes_of_face[0]);
    std::vector<int> pressure_reconstruction_cell_row_idx;
    pressure_reconstruction_cell_row_idx.reserve(grid.num_faces() * num_nodes_of_face[0]);
    std::vector<std::vector<int>> pressure_reconstruction_cell_col_idx;
    pressure_reconstruction_cell_col_idx.reserve(num_bound_faces * num_nodes_of_face[0]);
    // .. and for the face contributions.
    std::vector<std::vector<double>> pressure_reconstruction_face_values;
    pressure_reconstruction_face_values.reserve(num_bound_faces * num_nodes_of_face[0]);
    std::vector<int> pressure_reconstruction_face_row_idx;
    pressure_reconstruction_face_row_idx.reserve(grid.num_faces() * num_nodes_of_face[0]);
    std::vector<std::vector<int>> pressure_reconstruction_face_col_idx;
    pressure_reconstruction_face_col_idx.reserve(num_bound_faces * num_nodes_of_face[0]);

    // Data structures for the vector source terms.
    //
    // For the term representing imbalances in nK. Here we only need the values, since
    // the row and column indices can be inferred from the flux matrix.
    std::vector<std::vector<double>> vector_source_cell_values;
    vector_source_cell_values.reserve(grid.num_faces() * grid.dim() * num_nodes_of_face[0] + 1);

    // And for the face pressure reconstruction term (bound_pressure_vector_source).
    std::vector<std::vector<double>> vector_source_bound_pressure_values;
    vector_source_bound_pressure_values.reserve(
        num_bound_faces * grid.dim() * num_nodes_of_face[0] + 1);
    std::vector<int> vector_source_bound_pressure_row_idx;
    vector_source_bound_pressure_row_idx.reserve(
        grid.num_faces() * grid.dim() * num_nodes_of_face[0] + 1);
    std::vector<std::vector<int>> vector_source_bound_pressure_col_idx;
    vector_source_bound_pressure_col_idx.reserve(
        num_bound_faces * grid.dim() * num_nodes_of_face[0] + 1);

    const int DIM = grid.dim();
    int tot_num_transmissibilities = 0;

    std::vector<std::array<double, SPATIAL_DIM>> loc_cell_centers;
    std::vector<std::array<double, SPATIAL_DIM>> loc_face_centers;
    std::vector<std::array<double, SPATIAL_DIM>> loc_face_normals;

    // Storage for local (to interaction region) and global face indices of a cell.
    std::vector<int> loc_faces_of_cell(DIM, -1);
    std::vector<int> glob_faces_of_cell(DIM, -1);

    for (int node_ind{0}; node_ind < grid.num_nodes(); ++node_ind)
    {
        // Get the interaction region for the node.
        InteractionRegion interaction_region(node_ind, 1, grid);

        const int num_faces = interaction_region.faces().size();
        const int num_cells = interaction_region.cells().size();

        // Iterate over the matrix flux (columns major), store the values in the
        // flux_triplets.
        const auto& reg_cell_ind = interaction_region.cells();
        std::vector<int> reg_face_glob_ind;
        reg_face_glob_ind.reserve(num_faces);
        std::vector<int> reg_face_loc_ind;
        reg_face_loc_ind.reserve(num_faces);
        for (const auto& pair : interaction_region.faces())
        {
            reg_face_glob_ind.push_back(pair.first);
            reg_face_loc_ind.push_back(pair.second);
        }

        const std::vector<double> node_coord = grid.nodes()[node_ind];

        // Initialize matrices for the discretization.
        MatrixXd balance_cells = MatrixXd::Zero(num_faces, num_cells);
        MatrixXd balance_faces = MatrixXd::Zero(num_faces, num_faces);

        MatrixXd flux_cells = MatrixXd::Zero(num_faces, num_cells);
        MatrixXd flux_faces = MatrixXd::Zero(num_faces, num_faces);

        // Initialize the matrices used for the nK (vector source) terms.
        MatrixXd nK_matrix = MatrixXd::Zero(num_faces, SPATIAL_DIM * num_cells);
        MatrixXd nK_one_sided = MatrixXd::Zero(num_faces, SPATIAL_DIM * num_cells);

        // TODO: Should we use vectors for the inner quantities?
        loc_cell_centers.resize(num_cells);
        cell_centers_of_interaction_region(interaction_region, grid, loc_cell_centers);
        loc_face_centers.resize(num_faces);
        face_centers_of_interaction_region(interaction_region, grid, loc_face_centers);
        loc_face_normals.resize(num_faces);
        face_normals_of_interaction_region(interaction_region, grid, loc_face_normals);

        // If all cells have grid.dim() + 1 faces, this is a simplex. Use a boolean to
        // indicate whether this is a simplex or not.
        bool is_simplex = true;
        for (const int num : num_faces_of_cell)
        {
            if (num != (DIM + 1))
            {
                is_simplex = false;
                break;
            }
        }

        // Data structures to store the local boundary faces and their types.

        // Mapping from local face index to global face index.
        std::vector<std::pair<int, int>> loc_boundary_face_map;
        // Mapping from local face to an optional boundary condition type. If the face
        // is not on a boundary, contains `std::nullopt`.
        std::vector<std::optional<BoundaryCondition>> loc_boundary_faces_type(
            interaction_region.faces().size(), std::nullopt);
        std::map<int, std::vector<std::array<double, 3>>> basis_map;

        for (const auto& pair : interaction_region.faces())
        {
            // Initialize the local boundary faces with the local face index and the
            // global face index.
            auto it = bc_map.find(pair.first);
            if (it != bc_map.end())
            {
                BoundaryCondition bc = it->second;
                if (bc == BoundaryCondition::Dirichlet || bc == BoundaryCondition::Neumann)
                {
                    // Store the local face index for Neumann faces. We need to do
                    // some scaling of this in the boundary condition
                    // discretization.
                    // Store the local face index for Dirichlet or Neumann faces. We
                    // need to do some scaling of this in the boundary condition
                    // discretization.
                    loc_boundary_face_map.push_back({pair.second, pair.first});
                    loc_boundary_faces_type.at(pair.second) = bc;
                }
                // Other cases are not implemented.
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

        // Iterate over the faces in the interaction region.
        for (int loc_cell_ind{0}; loc_cell_ind < num_cells; ++loc_cell_ind)
        {
            continuity_points[0] = loc_cell_centers[loc_cell_ind];
            const int glob_cell_ind = reg_cell_ind[loc_cell_ind];

            int face_counter = 1;
            for (const int glob_face_ind : interaction_region.faces_of_cells().at(glob_cell_ind))
            {
                // Get the face normal and center.
                const int loc_face_index = interaction_region.faces().at(glob_face_ind);
                glob_faces_of_cell[face_counter - 1] = glob_face_ind;
                loc_faces_of_cell[face_counter - 1] = loc_face_index;

                bool in_dir = false, in_neu = false;
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
                            2.0 / 3 * loc_face_centers[loc_face_index][i] +
                            (1.0 / 3) * node_coord[i];
                    }
                }
                else
                {
                    for (int k{0}; k < SPATIAL_DIM; ++k)
                    {
                        continuity_points[face_counter][k] = loc_face_centers[loc_face_index][k];
                    }
                }
                ++face_counter;
            }
            basis_functions = basis_constructor.compute_basis_functions(continuity_points);

            for (int outer_face_counter{0}; outer_face_counter < loc_faces_of_cell.size();
                 ++outer_face_counter)
            {
                // Global and local face indices.
                const int glob_face_ind = glob_faces_of_cell[outer_face_counter];
                const int loc_face_index = loc_faces_of_cell[outer_face_counter];

                // Find the boundary condition for the face, if any.
                bool is_boundary_face = false;

                std::array<double, 3> flux_expr =
                    nK(loc_face_normals[loc_face_index], tensor, glob_cell_ind,
                       num_nodes_of_face[glob_face_ind]);

                // Here we need a map to the local flux index to get the right storage in the
                // matrices.
                const int sign = grid.sign_of_face_cell(glob_face_ind, glob_cell_ind);

                // We need the nK gradient for the flux expression, independent of
                // whether this is an internal or boundary face, and the type of
                // boundary condition.
                std::vector<double> flux_vals = nKgrad(flux_expr, basis_functions);

                // Decleare the vector for the Dirichlet values. This may or may not be
                // used in calculations below.
                std::vector<double> dirichlet_vals;

                is_boundary_face = false;
                BoundaryCondition bc;
                if (const auto optional_bc = loc_boundary_faces_type[loc_face_index];
                    optional_bc.has_value())
                {
                    is_boundary_face = true;
                    bc = *optional_bc;
                }

                // Note on the sign of the elements in the balance matrices: For
                // internal faces, where the balance equation describes the continuity
                // of flux expressions nKgrad p (the last factor is represented by basis
                // functions) for the two sides of a face, it is the unimportant whether
                // we move the cell or the face dependency to the left or right hand
                // side (that is, which of the two is multiplied with -1). For the
                // boundary faces, however, the balance equation consists of a one-sided
                // flux expression (for Neumann conditions) or a pressure continuity
                // condition (for Dirichlet), with each equated to the actual boundary
                // condition (write this up and do some thinking, it will make sense at
                // some point). In this case, we need to isolate the face values on one
                // side, and the cell and boundary condition on the other side. To that
                // end, it is convenient to let the flux values be scaled with 1, cell
                // values with -1.

                if (is_boundary_face && (bc == BoundaryCondition::Dirichlet))
                {
                    // For Dirichlet boundary conditions, the condition imposed for the
                    // local balance problem is one of pressure continuity.
                    dirichlet_vals = p_diff(loc_face_centers[loc_face_index],
                                            loc_cell_centers[loc_cell_ind], basis_functions);

                    // Note to self: There is no multiplication with sign here, since
                    // Dirichlet condition does not see the direction of the normal
                    // vector. The contribution to balance_cells is the pressure
                    // difference due to the cell center basis function
                    // (dirichlet_vals[0]) + 1, where the last term represents the
                    // offset from the cell center pressure.
                    balance_cells(loc_face_index, loc_cell_ind) = -dirichlet_vals[0] - 1.0;
                }
                else
                {
                    balance_cells(loc_face_index, loc_cell_ind) = -sign * flux_vals[0];

                    // Store the nK values in the nK_matrix for the vector source term.
                    // This is only necessary for Neumann and internal faces.
                    for (int i = 0; i < SPATIAL_DIM; ++i)
                    {
                        const int col = i + SPATIAL_DIM * loc_cell_ind;
                        nK_matrix(loc_face_index, col) = sign * flux_expr[i];
                    }
                }

                for (int i = 1; i < DIM + 1; ++i)
                {
                    const int loc_face_index_secondary = loc_faces_of_cell[i - 1];

                    // For the local balance problem, the condition imposed differs
                    // between on the one hand Dirichlet faces and on the other hand
                    // Neumann and internal faces.
                    //
                    // In both cases, the sign for the assignment to the balance_faces
                    // matrix, must be the opposite of the sign used in balance_cells,
                    // since we gather all contributions from faces and cells on
                    // different sides of an equation.
                    if (is_boundary_face && (bc == BoundaryCondition::Dirichlet))
                    {
                        balance_faces(loc_face_index, loc_face_index_secondary) +=
                            dirichlet_vals[i];
                    }
                    else
                    {
                        balance_faces(loc_face_index, loc_face_index_secondary) +=
                            sign * flux_vals[i];
                    }
                }

                // The discretization of the flux is the same for internal and boundary
                // faces; it is the nK product times the basis function for face and
                // cell.
                //
                // EK comment to self: No scaling with sign here. This is just a
                // computation of n * K * grad, with the gradient calculated according
                // to the geometry of the main cell, and n having the direction it has.
                // Interpretation/distribution of the flux as inwards or outwards is
                // left to the discrete divergence operator (elsewhere in the code).
                if (glob_cell_ind == interaction_region.main_cell_of_faces().at(loc_face_index))
                {
                    // If this is the main cell for the face, we store the flux in the
                    // balance_faces matrix.
                    flux_cells(loc_face_index, loc_cell_ind) = flux_vals[0];
                    for (int i = 1; i < DIM + 1; ++i)
                    {
                        const int glob_face_index_secondary =
                            interaction_region.faces_of_cells().at(glob_cell_ind)[i - 1];
                        const int loc_face_index_secondary =
                            interaction_region.faces().at(glob_face_index_secondary);

                        flux_faces(loc_face_index, loc_face_index_secondary) = flux_vals[i];
                    }

                    // Add to the one-sided nK values, unless this is a Dirichlet
                    // boundary.
                    if (~(is_boundary_face && (bc == BoundaryCondition::Dirichlet)))
                    {
                        for (int k = 0; k < SPATIAL_DIM; ++k)
                        {
                            // EK note to self: Not 100% sure about the reason for the
                            // factor -1 here, but it seems to be necessary to ensure
                            // equivalence with the PorePy discretization.
                            nK_one_sided(loc_face_index, k + loc_cell_ind * SPATIAL_DIM) =
                                -flux_expr[k];
                        }
                    }
                }

                // If this is a boundary face, we store the basis function so that we
                // can compute the boundary pressure reconstruction later.
                if (is_boundary_face)
                {
                    basis_map[glob_cell_ind] = basis_functions;
                }
            }
        }  // End iteration of cells of the interaction region.

        // Compute the inverse of balance_faces matrix.
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> balance_faces_inv;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flux;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bound_flux;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vector_source_cell;

        balance_faces_inv = balance_faces.inverse();
        bound_flux.noalias() = flux_faces * balance_faces_inv;

        bool has_dirichlet = false;
        for (const auto bc : loc_boundary_faces_type)
        {
            if (bc.has_value() && *bc == BoundaryCondition::Dirichlet)
            {
                has_dirichlet = true;
                break;
            }
        }

        if (!has_dirichlet)
        {
            // If there are no Dirichlet faces, we can directly use the flux_cells matrix.
            flux.noalias() = bound_flux * balance_cells + flux_cells;
        }
        else
        {
            // Create a mask representing a diagonal matrix which has value 0.0 for
            // Dirichlet faces and 1.0 for all other faces.
            // TODO: EK believes this also applies to Neumann faces. That should become
            // clear when applying this to a grid that is not K-orthogonal.
            Eigen::VectorXd mask = Eigen::VectorXd::Ones(num_faces);

            for (int face{0}; face < loc_boundary_faces_type.size(); ++face)
            {
                const auto bc = loc_boundary_faces_type[face];
                if (bc.has_value() && *bc == BoundaryCondition::Dirichlet)
                {
                    mask(face) = 0.0;  // Dirichlet faces
                }
            }

            flux.noalias() = bound_flux * mask.asDiagonal() * balance_cells + flux_cells;
        }

        // Matrix needed to compute the vector source term.
        vector_source_cell.noalias() = bound_flux * nK_matrix + nK_one_sided;

        // Store the computed flux in the flux_matrix_values, row_idx, and col_idx.
        size_t vs_cols = vector_source_cell.cols();
        size_t cols = flux.cols();
        for (const auto face_inds : interaction_region.faces())
        {
            long row_id = face_inds.second;

            // Constructing the vector IN PLACE at the end of the list. This avoids
            // creating a temporary vector and then moving/copying it.
            flux_matrix_values.emplace_back();
            auto& new_flux_row = flux_matrix_values.back();
            new_flux_row.resize(cols);

            // Copying raw data. We use RowMajor format, so data is contiguous.
            std::memcpy(new_flux_row.data(), flux.row(row_id).data(), cols * sizeof(double));

            flux_matrix_row_idx.push_back(face_inds.first);

            // Same for Vector Source.
            vector_source_cell_values.emplace_back();
            auto& new_vs_row = vector_source_cell_values.back();
            new_vs_row.resize(vs_cols);
            std::memcpy(new_vs_row.data(), vector_source_cell.row(row_id).data(),
                        vs_cols * sizeof(double));
        }
        // Store column indices for the flux matrix.
        for (int i = 0; i < num_faces; ++i)
        {
            flux_matrix_col_idx.emplace_back(interaction_region.cells());
        }

        tot_num_transmissibilities += num_faces * num_cells;

        if (loc_boundary_face_map.size() > 0)
        {
            // Also need to find the face pressures as generated by the nK imbalances; this
            // will enter the reconstruction of face boundary pressures from the vector
            // sources.
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
                bound_vector_source_matrix;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
                face_pressure_from_cells;
            bound_vector_source_matrix.noalias() = balance_faces_inv * nK_matrix;

            // Loop over all faces in the interaction region (internal and boundary).
            // Pick out the flux discretization associated with boundary faces (this can
            // be thought of as discretization of the flux induced by the boundary
            // condition).

            for (const auto& face_pair : interaction_region.faces())
            {
                // For the boundary faces, we need to compute the boundary flux matrix.

                bound_flux_matrix_row_idx.emplace_back(face_pair.first);
                // Creating the inner vectors in place to avoid additional copying into
                // them.
                bound_flux_matrix_col_idx.emplace_back();
                bound_flux_matrix_values.emplace_back();
                auto& bf_indices = bound_flux_matrix_col_idx.back();
                auto& bf_val = bound_flux_matrix_values.back();

                for (const auto& loc_face_pair : loc_boundary_face_map)
                {
                    if (const auto bc = loc_boundary_faces_type[loc_face_pair.first];
                        bc.has_value() && *bc == BoundaryCondition::Neumann)
                    {
                        // For Neumann boundary faces, scale the flux by the number of
                        // nodes, since the boundary condition will be taken in terms of the
                        // total flux over the face (not the subface).
                        bf_val.push_back(bound_flux(face_pair.second, loc_face_pair.first) /
                                         num_nodes_of_face[loc_face_pair.first]);
                    }
                    else
                    {
                        // We know it can be only Dirichlet.
                        bf_val.push_back(bound_flux(face_pair.second, loc_face_pair.first));
                    }
                    bf_indices.push_back(loc_face_pair.second);
                }
            }

            // Mapping from cell center pressure to face pressures.
            face_pressure_from_cells.noalias() = balance_faces_inv * balance_cells;

            // Vector of the global indices of the interaction region faces.
            std::vector<int> glob_indices_iareg_faces;
            for (const auto& face_pair : interaction_region.faces())
            {
                glob_indices_iareg_faces.push_back(face_pair.first);
            }

            for (const auto& loc_face_pair : loc_boundary_face_map)
            {
                // We need to divide by the number of nodes on the face, since the
                // pressure at the face is defined as the average of the pressure at the
                // nodes on the face.
                const double inv_num_nodes_of_face = 1.0 / num_nodes_of_face[loc_face_pair.first];

                if (const auto bc = loc_boundary_faces_type[loc_face_pair.first];
                    bc.has_value() && *bc == BoundaryCondition::Dirichlet)
                {
                    // For a Dirichlet boundary face, we only need to assign a unit
                    // value (thereby, the pressure at the face will be equal to the
                    // prescribed boundary condition).

                    // Creates a vector of one element in place.
                    pressure_reconstruction_face_values.emplace_back(
                        std::initializer_list<double>{1.0 * inv_num_nodes_of_face});
                    pressure_reconstruction_face_row_idx.push_back(loc_face_pair.second);
                    // Same in place construction.
                    pressure_reconstruction_face_col_idx.emplace_back(
                        std::initializer_list<int>{loc_face_pair.second});
                    // No contribution from other cells or faces.
                    continue;
                }

                // Find the cell next to the boundary face. This pressure at the
                // boundary face will be a perturbation from the value at this cell
                // center. A boundary face will per definition have a single cell
                // neighbor, which will be returned as the main cell for the face.
                const int glob_cell_ind =
                    interaction_region.main_cell_of_faces().at(loc_face_pair.first);
                // Identify the local index of the cell in the interaction region.
                const int loc_cell_ind =
                    std::find(interaction_region.cells().begin(), interaction_region.cells().end(),
                              glob_cell_ind) -
                    interaction_region.cells().begin();

                // Compute the pressure difference between the face center and the cell
                // center, using the set of basis functions for this cell.
                const std::vector<double> pressure_diff =
                    p_diff(loc_face_centers[loc_face_pair.first], loc_cell_centers[loc_cell_ind],
                           basis_map[glob_cell_ind]);

                // Cell contribution to pressure reconstruction.
                std::vector<double> cell_contribution(interaction_region.cells().size(), 0.0);
                // Cell contribution to vector source pressure reconstruction.
                std::vector<double> vector_source_cell_contribution(
                    interaction_region.cells().size() * SPATIAL_DIM, 0.0);

                // The cell itself contributes a unit value (which gives the offset) to
                // the cell + contribution from the gradient.
                cell_contribution[loc_cell_ind] = (1.0 + pressure_diff[0]) * inv_num_nodes_of_face;
                // EK note to self: There is no contribution cell-wise from the nK term
                // to the vector source; this all goes through the boundary faces below.
                // I'm not 100% convinced by the logic here, but this is what it takes
                // to be compatible with PorePy, so it will have to do for now.

                std::vector<double> face_contribution(interaction_region.faces().size(), 0.0);

                // Loop over the faces of the cell that also belong to the interaction
                // region. The pressure on each of these faces contributes to the
                // pressure variation within the cell.

                // Start at 1, since the first basis function is the cell center
                // pressure.
                int basis_vector_face_counter = 1;
                for (const int face_ind : interaction_region.faces_of_cells().at(glob_cell_ind))
                {
                    const int face_local_index = interaction_region.faces().at(face_ind);

                    // This row maps cell center pressures to the face pressure at face
                    // face_local_index.
                    std::vector<double> row_from_cells(
                        face_pressure_from_cells.row(face_local_index).data(),
                        face_pressure_from_cells.row(face_local_index).data() +
                            face_pressure_from_cells.row(face_local_index).size());

                    std::vector<double> row_from_cells_vector_source(
                        bound_vector_source_matrix.row(face_local_index).data(),
                        bound_vector_source_matrix.row(face_local_index).data() +
                            bound_vector_source_matrix.row(face_local_index).size());

                    for (int loc_cell_ind{0}; loc_cell_ind < interaction_region.cells().size();
                         ++loc_cell_ind)
                    {
                        // If the cell is not the main cell for the face, we need to
                        // multiply the pressure difference with the basis function for
                        // this cell.
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
                            // Get the contribution from face_pair.first via the basis
                            // function centered at the face face_ind. Rough explanation
                            // of the double factor inv_num_nodes_of_face: One comes
                            // from dividing the boundary condition (with which the
                            // boundary contribution will be scaled) by the number of
                            // nodes of the face (think, divide the total flux by the
                            // number of nodes, to get the average flux per subface).
                            // The second reflects the averaging of the reconstructed
                            // face pressure at the boundary face. NOTE: The first factor
                            // inv_num_nodes_of_face should really be on faces.first,
                            // not face.second (we want to scale the boundary condition
                            // on the right face), but we assume all faces have the same
                            // number of nodes.

                            // Divide by the number of nodes of the face, to get an
                            // averaged face pressure (over the contribution from the
                            // subfaces).
                            double contribution_from_face =
                                row_from_faces[loc_face_ind] *
                                pressure_diff[basis_vector_face_counter] * inv_num_nodes_of_face;

                            if (*bc == BoundaryCondition::Neumann)
                            {
                                // For Neumann faces, we also need to divide the imposed
                                // flux by the number of nodes. This is equivalent to
                                // the scaling in bound_flux_matrix_values for Neumann
                                // boundaries.
                                contribution_from_face *= inv_num_nodes_of_face;
                            }

                            face_contribution[loc_face_ind] += contribution_from_face;
                        }
                    }
                    ++basis_vector_face_counter;
                }
                pressure_reconstruction_cell_values.emplace_back(cell_contribution);
                pressure_reconstruction_cell_row_idx.push_back(loc_face_pair.second);
                pressure_reconstruction_cell_col_idx.emplace_back(interaction_region.cells());
                pressure_reconstruction_face_values.emplace_back(face_contribution);
                pressure_reconstruction_face_row_idx.push_back(loc_face_pair.second);
                pressure_reconstruction_face_col_idx.emplace_back(glob_indices_iareg_faces);

                vector_source_bound_pressure_values.emplace_back(vector_source_cell_contribution);
                vector_source_bound_pressure_row_idx.push_back(loc_face_pair.second);

                std::vector<int> cell_ind_vector_source;
                for (auto& loc_cell_ind : interaction_region.cells())
                {
                    for (int k = 0; k < SPATIAL_DIM; ++k)
                    {
                        const int col = loc_cell_ind * SPATIAL_DIM + k;
                        cell_ind_vector_source.push_back(col);
                    }
                }
                vector_source_bound_pressure_col_idx.emplace_back(cell_ind_vector_source);
            }
        }
    }  // End iteration of nodes in the grid.

    // Gather the computed data into CSR matrices and further into the discretization
    // structure.
    ScalarDiscretization discretization;

    // Use a tailored method for creating CSR matrices for the flux and vector source
    // matrices, since they have similar sparsity pattern, and the construction of these
    // take considerable time.
    auto flux_and_vs = create_flux_vector_source_matrix(
        flux_matrix_row_idx, flux_matrix_col_idx, flux_matrix_values, vector_source_cell_values,
        grid.num_faces(), grid.num_cells(), tot_num_transmissibilities);
    discretization.flux = flux_and_vs.first;
    discretization.vector_source = flux_and_vs.second;

    // For the remaining matrices we use a more general method. There could be some
    // savings to be gained here as well, but in the bigger picture, the time spent
    // here is not that important.
    auto bound_flux_storage = create_csr_matrix(
        bound_flux_matrix_row_idx, bound_flux_matrix_col_idx, bound_flux_matrix_values,
        grid.num_faces(), grid.num_faces(), bound_flux_matrix_row_idx.size());
    discretization.bound_flux = bound_flux_storage;
    auto pressure_reconstruction_cell_storage = create_csr_matrix(
        pressure_reconstruction_cell_row_idx, pressure_reconstruction_cell_col_idx,
        pressure_reconstruction_cell_values, grid.num_faces(), grid.num_cells(),
        pressure_reconstruction_cell_row_idx.size());
    discretization.bound_pressure_cell = pressure_reconstruction_cell_storage;
    auto pressure_reconstruction_face_storage = create_csr_matrix(
        pressure_reconstruction_face_row_idx, pressure_reconstruction_face_col_idx,
        pressure_reconstruction_face_values, grid.num_faces(), grid.num_faces(),
        pressure_reconstruction_face_row_idx.size());
    discretization.bound_pressure_face = pressure_reconstruction_face_storage;
    auto vector_source_bound_pressure_storage = create_csr_matrix(
        vector_source_bound_pressure_row_idx, vector_source_bound_pressure_col_idx,
        vector_source_bound_pressure_values, grid.num_faces(), SPATIAL_DIM * grid.num_cells(),
        vector_source_bound_pressure_row_idx.size());
    discretization.bound_pressure_vector_source = vector_source_bound_pressure_storage;

    return discretization;
}