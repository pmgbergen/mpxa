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

const std::vector<double> nK(const std::vector<double>& face_normal,
                             const SecondOrderTensor& tensor, const int cell_ind,
                             const int num_nodes_of_face)
{
    // Compute the product between the normal vector, the tensor, and the cell-face
    // vector.

    const int dim = face_normal.size();
    std::vector<double> result(dim, 0.0);
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
std::vector<std::vector<double>> cell_centers_of_interaction_region(
    const InteractionRegion& interaction_region, const Grid& grid)
{
    std::vector<std::vector<double>> centers;
    for (const int cell_ind : interaction_region.cells())
    {
        centers.push_back(grid.cell_center(cell_ind));
    }
    return centers;
}

// Helper function to get face center coordinates for all faces in an interaction region
std::vector<std::vector<double>> face_centers_of_interaction_region(
    const InteractionRegion& interaction_region, const Grid& grid)
{
    std::vector<std::vector<double>> centers;
    for (const auto& pair : interaction_region.faces())
    {
        centers.push_back(grid.face_center(pair.first));
    }
    return centers;
}

// Helper function to get face normals for all faces in an interaction region
std::vector<std::vector<double>> face_normals_of_interaction_region(
    const InteractionRegion& interaction_region, const Grid& grid)
{
    std::vector<std::vector<double>> normals;
    for (const auto& pair : interaction_region.faces())
    {
        normals.push_back(grid.face_normal(pair.first));
    }
    return normals;
}

std::vector<double> nKgrad(const std::vector<double>& nK,
                           const std::vector<std::vector<double>>& basis_functions)
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

std::vector<double> p_diff(const std::vector<double>& face_center,
                           const std::vector<double>& cell_center,
                           const std::vector<std::vector<double>>& basis_functions)
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
    const std::vector<std::vector<double>>& flux_matrix_values, const Grid& grid,
    const int tot_num_transmissibilities)
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
    while (row_ptr.size() <= grid.num_faces())
    {
        row_ptr.push_back(flux_values.size());
    }

    // Create the compressed data storage for the flux.
    auto flux_storage = std::make_shared<CompressedDataStorage<double>>(
        grid.num_faces(), grid.num_cells(), row_ptr, col_idx, flux_values);
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
    std::vector<std::vector<double>> continuty_points(grid.dim() + 1,
                                                      std::vector<double>(SPATIAL_DIM, 0.0));

    std::vector<std::vector<double>> basis_functions(grid.dim() + 1,
                                                     std::vector<double>(SPATIAL_DIM, 0.0));

    // Data structures for the computed stencils.
    std::vector<std::vector<double>> flux_matrix_values;
    std::vector<int> flux_matrix_row_idx;
    std::vector<std::vector<int>> flux_matrix_col_idx;

    // Data structures for the discretization of boundary conditions.
    std::vector<std::vector<double>> bound_flux_matrix_values;
    std::vector<int> bound_flux_matrix_row_idx;
    std::vector<std::vector<int>> bound_flux_matrix_col_idx;

    const int DIM = grid.dim();
    int tot_num_transmissibilities = 0;

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

        // TODO: Should we use vectors for the inner quantities?
        std::vector<std::vector<double>> loc_cell_centers =
            cell_centers_of_interaction_region(interaction_region, grid);
        std::vector<std::vector<double>> loc_face_centers =
            face_centers_of_interaction_region(interaction_region, grid);
        std::vector<std::vector<double>> loc_face_normals =
            face_normals_of_interaction_region(interaction_region, grid);

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
        std::vector<std::pair<int, int>> loc_boundary_faces;
        std::unordered_set<int> loc_neumann_faces;
        std::unordered_set<int> loc_dirichlet_faces;
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

                loc_boundary_faces.push_back({pair.second, pair.first});
            }
        }

        // Iterate over the faces in the interaction region.
        for (int loc_cell_ind{0}; loc_cell_ind < num_cells; ++loc_cell_ind)
        {
            continuty_points[0] = loc_cell_centers[loc_cell_ind];
            const int cell_ind = interaction_region.cells()[loc_cell_ind];

            std::vector<int> loc_faces_of_cell;
            std::vector<int> glob_faces_of_cell;

            int face_counter = 1;
            for (const int face_ind : interaction_region.faces_of_cells().at(cell_ind))
            {
                // Get the face normal and center.
                const int local_face_index = interaction_region.faces().at(face_ind);
                glob_faces_of_cell.push_back(face_ind);
                loc_faces_of_cell.push_back(local_face_index);

                auto in_dir = loc_dirichlet_faces.find(local_face_index);
                auto in_neu = loc_neumann_faces.find(local_face_index);

                if (is_simplex && (in_dir == loc_dirichlet_faces.end()) &&
                    (in_neu == loc_neumann_faces.end()))
                {
                    for (int i = 0; i < SPATIAL_DIM; ++i)
                    {
                        continuty_points[face_counter][i] =
                            2.0 / 3 * loc_face_centers[local_face_index][i] +
                            (1.0 / 3) * node_coord[i];
                    }
                }
                else
                {
                    continuty_points[face_counter] = loc_face_centers[local_face_index];
                }
                ++face_counter;
            }
            basis_functions = basis_constructor.compute_basis_functions(continuty_points);

            for (int outer_face_counter{0}; outer_face_counter < loc_faces_of_cell.size();
                 ++outer_face_counter)
            {
                // Global and local face indices.
                const int face_ind = glob_faces_of_cell[outer_face_counter];
                const int local_face_index = loc_faces_of_cell[outer_face_counter];

                // Find the boundary condition for the face, if any.
                bool is_boundary_face = false;

                std::vector<double> flux_expr = nK(loc_face_normals[local_face_index], tensor,
                                                   cell_ind, num_nodes_of_face.at(face_ind));

                // Here we need a map to the local flux index to get the right storage in the
                // matrices.
                const int sign = grid.sign_of_face_cell(face_ind, cell_ind);

                // We need the nK gradient for the flux expression, independent of
                // whether this is an internal or boundary face, and the type of
                // boundary condition.
                std::vector<double> flux_vals = nKgrad(flux_expr, basis_functions);
                // Decleare the vector for the Dirichlet values. This may or may not be
                // used in calculations below.
                std::vector<double> dirichlet_vals;

                is_boundary_face = false;
                BoundaryCondition bc;
                if (bc_map.find(face_ind) != bc_map.end())
                {
                    is_boundary_face = true;
                    bc = bc_map.at(face_ind);
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
                    dirichlet_vals = p_diff(loc_face_centers[local_face_index],
                                            loc_cell_centers[loc_cell_ind], basis_functions);

                    // Note to self: There is no multiplication with sign here, since
                    // the Dirichlet condition does not see the direction of the normal
                    // vector.
                    balance_cells(local_face_index, loc_cell_ind) = -dirichlet_vals[0];
                }
                else
                {
                    balance_cells(local_face_index, loc_cell_ind) = -sign * flux_vals[0];
                }

                for (int i = 1; i < DIM + 1; ++i)
                {
                    const int face_index_secondary = glob_faces_of_cell[i - 1];
                    const int face_index_secondary_local = loc_faces_of_cell[i - 1];

                    // For the local balance problem, the condition imposed differs
                    // between on the one hand Dirichlet faces and on the other hand
                    // Neumann and internal faces.
                    //
                    // In both cases, the for the assignment to the balance_faces
                    // matrix, must be the opposite of the sign used in balance_cells,
                    // since we gather all contributions from faces and cells on
                    // different sides of an equation.
                    if (is_boundary_face && (bc == BoundaryCondition::Dirichlet))
                    {
                        balance_faces(local_face_index, face_index_secondary_local) +=
                            dirichlet_vals[i];
                    }
                    else
                    {
                        balance_faces(local_face_index, face_index_secondary_local) +=
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
                if (cell_ind == interaction_region.main_cell_of_faces().at(local_face_index))
                {
                    // If this is the main cell for the face, we store the flux in the
                    // balance_faces matrix.
                    flux_cells(local_face_index, loc_cell_ind) = flux_vals[0];
                    for (int i = 1; i < DIM + 1; ++i)
                    {
                        const int face_index_secondary =
                            interaction_region.faces_of_cells().at(cell_ind)[i - 1];
                        const int face_index_secondary_local =
                            interaction_region.faces().at(face_index_secondary);

                        flux_faces(local_face_index, face_index_secondary_local) = flux_vals[i];
                    }
                }
            }
        }  // End iteration of cells of the interaction region.

        // Compute the inverse of balance_faces matrix.
        MatrixXd balance_faces_inv = balance_faces.inverse();
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

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmp;
        tmp = bound_flux * balance_cells;

        // Store the computed flux in the flux_matrix_values, row_idx, and col_idx.
        for (const auto i : interaction_region.faces())
        {
            std::vector<double> row(flux.row(i.second).data(),
                                    flux.row(i.second).data() + flux.cols());
            flux_matrix_values.push_back(row);

            flux_matrix_row_idx.push_back(i.first);
            flux_matrix_col_idx.push_back(interaction_region.cells());
        }
        tot_num_transmissibilities += num_faces * num_cells;

        for (const auto& face : loc_boundary_faces)
        {
            // For the boundary faces, we need to compute the boundary flux matrix.
            std::vector<double> row(bound_flux.row(face.first).data(),
                                    bound_flux.row(face.first).data() + bound_flux.cols());

            std::vector<int> bf_indices;
            std::vector<double> bf_val;

            for (const auto& f : loc_boundary_faces)
            {
                if (loc_neumann_faces.find(f.first) != loc_neumann_faces.end())
                {
                    // For Neumann boundary faces, scale the flux by the number of
                    // nodes, since the boundary condition will be taken in terms of the
                    // total flux over the face (not the subface).
                    bf_val.push_back(row[f.first] / num_nodes_of_face.at(f.second));
                }
                else
                {
                    bf_val.push_back(row[f.first]);
                }
                bf_indices.push_back(f.second);
            }
            bound_flux_matrix_values.push_back(bf_val);
            bound_flux_matrix_col_idx.push_back(bf_indices);
            bound_flux_matrix_row_idx.push_back(face.second);
        }

    }  // End iteration of nodes in the grid.

    ScalarDiscretization discretization;

    // CSR storage for the flux matrix.
    auto flux_storage = create_csr_matrix(flux_matrix_row_idx, flux_matrix_col_idx,
                                          flux_matrix_values, grid, tot_num_transmissibilities);
    discretization.flux = flux_storage;
    // Create the compressed data storage for the boundary flux.
    auto bound_flux_storage =
        create_csr_matrix(bound_flux_matrix_row_idx, bound_flux_matrix_col_idx,
                          bound_flux_matrix_values, grid, bound_flux_matrix_row_idx.size());
    discretization.bound_flux = bound_flux_storage;
    return discretization;
}