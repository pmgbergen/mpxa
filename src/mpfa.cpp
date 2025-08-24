#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <map>
#include <numeric>
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
        std::vector<double> diag = tensor.diagonal_data(cell_ind);
        for (int i{0}; i < dim; ++i)
        {
            result[i] = -face_normal[i] * diag[i] * num_nodes_of_face_inv;
        }
    }
    else
    {
        std::vector<double> full_data = tensor.full_data(cell_ind);
        for (int i{0}; i < dim; ++i)
        {
            double tensor_val;
            for (int j{0}; j < dim; ++j)
            {
                if (i == 0 && j == 0)
                    tensor_val = full_data[0];
                else if (i == 1 && j == 1)
                    tensor_val = full_data[1];
                else if (i == 2 && j == 2)
                    tensor_val = full_data[2];
                else if (i == 0 && j == 1 || i == 1 && j == 0)
                    tensor_val = full_data[3];
                else if (i == 0 && j == 2 || i == 2 && j == 0)
                    tensor_val = full_data[4];
                else if (i == 1 && j == 2 || i == 2 && j == 1)
                    tensor_val = full_data[5];
                // TODO: Check i and j indices for correctness.
                result[i] -= face_normal[j] * tensor_val * num_nodes_of_face_inv;
            }
        }
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

std::map<int, int> count_nodes_of_faces(const InteractionRegion& interaction_region,
                                        const Grid& grid)
{
    // Count the number of nodes for each face in the interaction region.
    std::map<int, int> num_nodes_of_face;
    for (const auto& face : interaction_region.faces())
    {
        num_nodes_of_face[face.first] = grid.num_nodes_of_face(face.first);
    }
    return num_nodes_of_face;
}

std::vector<int> count_faces_of_cells(const InteractionRegion& interaction_region, const Grid& grid)
{
    // Count the number of faces for each cell in the interaction region.
    std::vector<int> num_faces_of_cell;
    for (const int cell : interaction_region.cells())
    {
        num_faces_of_cell.push_back(grid.faces_of_cell(cell).size());
    }
    return num_faces_of_cell;
}

// Helper function to create a compressed sparse row (CSR) matrix from vectors.
std::shared_ptr<CompressedDataStorage<double>> create_csr_matrix(
    const std::vector<int>& flux_matrix_row_idx,
    const std::vector<std::vector<int>>& flux_matrix_col_idx,
    const std::vector<std::vector<double>>& flux_matrix_values, const int num_rows,
    const int num_cols, const int tot_num_transmissibilities)
{
    std::vector<int> sorted_indices(flux_matrix_row_idx.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&flux_matrix_row_idx](int i1, int i2)
              { return flux_matrix_row_idx[i1] < flux_matrix_row_idx[i2]; });

    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> flux_values;
    col_idx.reserve(tot_num_transmissibilities);
    flux_values.reserve(tot_num_transmissibilities);

    int previous_row = -1;

    std::map<int, double> flux_matrix_values_map;

    // Loop over the sorted indices, which correspond to the subfaces.
    for (const int index : sorted_indices)
    {
        // If the row index has changed, that is, we have reached a new face, we need to
        // update the row_ptr.
        if (flux_matrix_row_idx[index] != previous_row)
        {
            // We need to store the flux values for the row, and empty the map for the next
            // row.
            for (const auto& pair : flux_matrix_values_map)
            {
                col_idx.push_back(pair.first);
                flux_values.push_back(pair.second);
            }
            flux_matrix_values_map.clear();

            // If we have a new row, we need to update the row_ptr. If there are zero
            // rows in the matrix, multiple values of row_ptr will be added.
            while (previous_row < flux_matrix_row_idx[index])
            {
                // We push the current size of col_idx to row_ptr.
                row_ptr.push_back(flux_values.size());
                ++previous_row;
            }
        }

        int counter = 0;
        for (const double col_index : flux_matrix_col_idx[index])
        {
            // Store the flux values in a map to gather duplicate column indices (would
            // correspond to the same face-cell combination being present in different
            // interaction regions).
            flux_matrix_values_map[col_index] += flux_matrix_values[index][counter];
            ++counter;
        }
    }
    // After the loop, we need to empty the map for the last non-zero row.
    for (const auto& pair : flux_matrix_values_map)
    {
        col_idx.push_back(pair.first);
        flux_values.push_back(pair.second);
    }
    // We also need to fill the row pointer for the last row. This may need to be
    // repeated, if the last non-zero row is not the last row in the matrix, hence the
    // while loop.
    while (row_ptr.size() <= num_rows)
    {
        row_ptr.push_back(flux_values.size());
    }

    // Create the compressed data storage for the flux.
    auto flux_storage = std::make_shared<CompressedDataStorage<double>>(num_rows, num_cols, row_ptr,
                                                                        col_idx, flux_values);
    return flux_storage;
}

}  // namespace

ScalarDiscretization mpfa(const Grid& grid, const SecondOrderTensor& tensor,
                          const std::map<int, BoundaryCondition>& bc_map)
{
    constexpr int SPATIAL_DIM = 3;  // Assuming 3D for now, can be generalized later.

    BasisConstructor basis_constructor(grid.dim());
    // Data structures for the discretization process. There will be grid.dim() continuity points
    // (outer vector).
    std::vector<std::array<double, 3>> continuity_points(grid.dim() + 1,
                                                         std::array<double, 3>{0.0, 0.0, 0.0});

    std::vector<std::array<double, 3>> basis_functions(grid.dim() + 1,
                                                       std::array<double, 3>{0.0, 0.0, 0.0});

    // Data structures for the computed stencils.
    std::vector<std::vector<double>> flux_matrix_values;
    std::vector<int> flux_matrix_row_idx;
    std::vector<std::vector<int>> flux_matrix_col_idx;

    // Data structures for the discretization of boundary conditions.
    std::vector<std::vector<double>> bound_flux_matrix_values;
    std::vector<int> bound_flux_matrix_row_idx;
    std::vector<std::vector<int>> bound_flux_matrix_col_idx;

    // Data structures for pressure reconstruction on boundary faces. Cell contributions.
    std::vector<std::vector<double>> pressure_reconstruction_cell_values;
    std::vector<int> pressure_reconstruction_cell_row_idx;
    std::vector<std::vector<int>> pressure_reconstruction_cell_col_idx;
    // .. and for the face contributions.
    std::vector<std::vector<double>> pressure_reconstruction_face_values;
    std::vector<int> pressure_reconstruction_face_row_idx;
    std::vector<std::vector<int>> pressure_reconstruction_face_col_idx;

    // Data structures for the vector source terms.
    //
    // For the term representing imbalances in nK.
    std::vector<std::vector<double>> vector_source_cell_values;
    std::vector<int> vector_source_cell_row_idx;
    std::vector<std::vector<int>> vector_source_cell_col_idx;
    // And for the face pressure reconstruction term (bound_pressure_vector_source).
    std::vector<std::vector<double>> vector_source_bound_pressure_values;
    std::vector<int> vector_source_bound_pressure_row_idx;
    std::vector<std::vector<int>> vector_source_bound_pressure_col_idx;

    const int DIM = grid.dim();
    int tot_num_transmissibilities = 0;

    std::vector<std::array<double, SPATIAL_DIM>> loc_cell_centers;
    std::vector<std::array<double, SPATIAL_DIM>> loc_face_centers;
    std::vector<std::array<double, SPATIAL_DIM>> loc_face_normals;

    for (int node_ind{0}; node_ind < grid.num_nodes(); ++node_ind)
    {
        // Get the interaction region for the node.
        InteractionRegion interaction_region(node_ind, 1, grid);

        const std::vector<double> node_coord = grid.nodes()[node_ind];

        const int num_faces = interaction_region.faces().size();
        const int num_cells = interaction_region.cells().size();

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

        std::map<int, int> num_nodes_of_face = count_nodes_of_faces(interaction_region, grid);
        std::vector<int> num_faces_of_cell = count_faces_of_cells(interaction_region, grid);

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
        // Sets to store the local boundary faces, Neumann faces, and Dirichlet faces.
        std::unordered_set<int> loc_boundary_faces;
        std::unordered_set<int> loc_neumann_faces;
        std::unordered_set<int> loc_dirichlet_faces;
        std::map<int, std::vector<std::array<double, 3>>> basis_map;

        for (const auto& pair : interaction_region.faces())
        {
            // Initialize the local boundary faces with the local face index and the
            // global face index.
            BoundaryCondition bc;
            auto it = bc_map.find(pair.first);
            if (it != bc_map.end())
            {
                bc = it->second;
                if (bc == BoundaryCondition::Neumann)
                {
                    // Store the local face index for Neumann faces. We need to do
                    // some scaling of this in the boundary condition
                    // discretization.
                    loc_neumann_faces.insert(pair.second);
                }
                if (bc == BoundaryCondition::Dirichlet)
                {
                    // Store the local face index for Dirichlet faces. We need to do
                    // some scaling of this in the boundary condition
                    // discretization.
                    loc_dirichlet_faces.insert(pair.second);
                }
                if (bc == BoundaryCondition::Robin)
                {
                    throw std::logic_error("Robin boundary condition not implemented");
                }

                loc_boundary_face_map.push_back({pair.second, pair.first});
                loc_boundary_faces.insert(pair.second);
            }
        }

        // Iterate over the faces in the interaction region.
        for (int loc_cell_ind{0}; loc_cell_ind < num_cells; ++loc_cell_ind)
        {
            continuity_points[0] = loc_cell_centers[loc_cell_ind];
            const int glob_cell_ind = interaction_region.cells()[loc_cell_ind];

            std::vector<int> loc_faces_of_cell;
            std::vector<int> glob_faces_of_cell;

            int face_counter = 1;
            for (const int glob_face_ind : interaction_region.faces_of_cells().at(glob_cell_ind))
            {
                // Get the face normal and center.
                const int loc_face_index = interaction_region.faces().at(glob_face_ind);
                glob_faces_of_cell.push_back(glob_face_ind);
                loc_faces_of_cell.push_back(loc_face_index);

                auto in_dir = loc_dirichlet_faces.find(loc_face_index);
                auto in_neu = loc_neumann_faces.find(loc_face_index);

                if (is_simplex && (in_dir == loc_dirichlet_faces.end()) &&
                    (in_neu == loc_neumann_faces.end()))
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
                       num_nodes_of_face.at(glob_face_ind));

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
                if (bc_map.find(glob_face_ind) != bc_map.end())
                {
                    is_boundary_face = true;
                    bc = bc_map.at(glob_face_ind);
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
        balance_faces_inv = balance_faces.inverse();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bound_flux;
        bound_flux = flux_faces * balance_faces_inv;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flux;
        if (loc_dirichlet_faces.empty())
        {
            // If there are no Dirichlet faces, we can directly use the flux_cells matrix.
            flux = bound_flux * balance_cells + flux_cells;
        }
        else
        {
            // Create a diagonal matrix which has value 0.0 for Dirichlet faces and 1.0 for
            // all other faces.
            // TODO: EK believes this also applies to Neumann faces. That should become
            // clear when applying this to a grid that is not K-orthogonal.
            Eigen::MatrixXd diag_matrix = Eigen::MatrixXd::Identity(num_faces, num_faces);
            for (const auto& face : loc_dirichlet_faces)
            {
                diag_matrix(face, face) = 0.0;  // Dirichlet faces
            }

            flux = bound_flux * diag_matrix * balance_cells + flux_cells;
        }

        // Matrix needed to compute the vector source term.
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vector_source_cell;
        vector_source_cell = bound_flux * nK_matrix + nK_one_sided;
        // Also need to find the face pressures as generated by the nK imbalances; this
        // will enter the reconstruction of face boundary pressures from the vector
        // sources.
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            bound_vector_source_matrix;
        bound_vector_source_matrix = balance_faces_inv * nK_matrix;

        // Store the computed flux in the flux_matrix_values, row_idx, and col_idx.
        for (const auto face_inds : interaction_region.faces())
        {
            std::vector<double> row(flux.row(face_inds.second).data(),
                                    flux.row(face_inds.second).data() + flux.cols());
            flux_matrix_values.push_back(row);

            flux_matrix_row_idx.push_back(face_inds.first);
            flux_matrix_col_idx.push_back(interaction_region.cells());

            // Also treatment of the vector source terms.
            std::vector<double> vs_row(
                vector_source_cell.row(face_inds.second).data(),
                vector_source_cell.row(face_inds.second).data() + vector_source_cell.cols());
            vector_source_cell_values.push_back(vs_row);
            vector_source_cell_row_idx.push_back(face_inds.first);

            std::vector<int> cell_indices;
            for (const auto& loc_cell_ind : interaction_region.cells())
            {
                for (int k = 0; k < SPATIAL_DIM; ++k)
                {
                    cell_indices.push_back(loc_cell_ind * SPATIAL_DIM + k);
                }
            }
            vector_source_cell_col_idx.push_back(cell_indices);
        }
        tot_num_transmissibilities += num_faces * num_cells;

        if (loc_boundary_face_map.size() > 0)
        {
            // Loop over all faces in the interaction region (internal and boundary).
            // Pick out the flux discretization associated with boundary faces (this can
            // be thought of as discretization of the flux induced by the boundary
            // condition).
            for (const auto& face_pair : interaction_region.faces())
            {
                // For the boundary faces, we need to compute the boundary flux matrix.
                std::vector<double> row(
                    bound_flux.row(face_pair.second).data(),
                    bound_flux.row(face_pair.second).data() + bound_flux.cols());

                std::vector<int> bf_indices;
                std::vector<double> bf_val;

                for (const auto& loc_face_pair : loc_boundary_face_map)
                {
                    if (loc_neumann_faces.find(loc_face_pair.first) != loc_neumann_faces.end())
                    {
                        // For Neumann boundary faces, scale the flux by the number of
                        // nodes, since the boundary condition will be taken in terms of the
                        // total flux over the face (not the subface).
                        bf_val.push_back(row[loc_face_pair.first] /
                                         num_nodes_of_face.at(loc_face_pair.second));
                    }
                    else
                    {
                        bf_val.push_back(row[loc_face_pair.first]);
                    }
                    bf_indices.push_back(loc_face_pair.second);
                }

                bound_flux_matrix_values.push_back(bf_val);
                bound_flux_matrix_col_idx.push_back(bf_indices);
                bound_flux_matrix_row_idx.push_back(face_pair.first);
            }

            // Mapping from cell center pressure to face pressures.
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
                face_pressure_from_cells;
            face_pressure_from_cells = balance_faces_inv * balance_cells;

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
                const double inv_num_nodes_of_face =
                    1.0 / num_nodes_of_face.at(loc_face_pair.second);

                if (loc_dirichlet_faces.find(loc_face_pair.first) != loc_dirichlet_faces.end())
                {
                    // For a Dirichlet boundary face, we only need to assign a unit
                    // value (thereby, the pressure at the face will be equal to the
                    // prescribed boundary condition).
                    std::vector<double> one{1.0 * inv_num_nodes_of_face};
                    pressure_reconstruction_face_values.push_back(one);
                    pressure_reconstruction_face_row_idx.push_back(loc_face_pair.second);
                    std::vector<int> col_idx{loc_face_pair.second};
                    pressure_reconstruction_face_col_idx.push_back(col_idx);
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
                        if (loc_boundary_faces.find(loc_face_ind) != loc_boundary_faces.end())
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
                            if (loc_neumann_faces.find(loc_face_ind) != loc_neumann_faces.end())
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
                pressure_reconstruction_cell_values.push_back(cell_contribution);
                pressure_reconstruction_cell_row_idx.push_back(loc_face_pair.second);
                pressure_reconstruction_cell_col_idx.push_back(interaction_region.cells());
                pressure_reconstruction_face_values.push_back(face_contribution);
                pressure_reconstruction_face_row_idx.push_back(loc_face_pair.second);
                pressure_reconstruction_face_col_idx.push_back(glob_indices_iareg_faces);

                vector_source_bound_pressure_values.push_back(vector_source_cell_contribution);
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

                vector_source_bound_pressure_col_idx.push_back(cell_ind_vector_source);
            }
        }

    }  // End iteration of nodes in the grid.

    ScalarDiscretization discretization;

    // CSR storage for the flux matrix.
    auto flux_storage =
        create_csr_matrix(flux_matrix_row_idx, flux_matrix_col_idx, flux_matrix_values,
                          grid.num_faces(), grid.num_cells(), tot_num_transmissibilities);
    discretization.flux = flux_storage;

    // Create the compressed data storage for the boundary flux.
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

    auto vector_source_cell_storage = create_csr_matrix(
        vector_source_cell_row_idx, vector_source_cell_col_idx, vector_source_cell_values,
        grid.num_faces(), SPATIAL_DIM * grid.num_cells(), vector_source_cell_row_idx.size());
    discretization.vector_source = vector_source_cell_storage;

    discretization.bound_pressure_vector_source = create_csr_matrix(
        vector_source_bound_pressure_row_idx, vector_source_bound_pressure_col_idx,
        vector_source_bound_pressure_values, grid.num_faces(), SPATIAL_DIM * grid.num_cells(),
        vector_source_bound_pressure_row_idx.size());

    return discretization;
}