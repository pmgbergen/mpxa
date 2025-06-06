#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <map>
#include <numeric>
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

    std::cerr << "Grid dimension: " << grid.dim() << "\n";
    // Data structures for the computed stencils.
    std::vector<std::vector<double>> flux_matrix_values;
    std::vector<int> flux_matrix_row_idx;
    std::vector<std::vector<int>> flux_matrix_col_idx;

    const int DIM = grid.dim();
    int tot_num_transmissibilities = 0;

    for (int node_ind{0}; node_ind < grid.num_nodes(); ++node_ind)
    {
        std::cerr << "Processing node " << node_ind << " of " << grid.num_nodes() << "\n";
        // Get the interaction region for the node.
        InteractionRegion interaction_region(node_ind, 1, grid);

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

        // Iterate over the faces in the interaction region.
        for (int loc_cell_ind{0}; loc_cell_ind < num_cells; ++loc_cell_ind)
        {
            continuty_points[0] = loc_cell_centers[loc_cell_ind];
            std::cerr << "Gathering face data for cell " << loc_cell_ind << "\n";
            const int cell_ind = interaction_region.cells()[loc_cell_ind];

            int face_counter = 1;
            for (const int face_ind : interaction_region.faces_of_cells().at(cell_ind))
            {
                // std::cerr << "Processing face " << face_ind << " of cell " << cell_ind << "\n";
                // Get the face normal and center.
                const int local_face_index = interaction_region.faces().at(face_ind);
                // std::cerr << "Local face index: " << local_face_index << "\n";

                // std::cerr << "Local face center: " << loc_face_centers[local_face_index][0] << ",
                // "
                //           << loc_face_centers[local_face_index][1] << ", "
                //           << loc_face_centers[local_face_index][2] << "\n";

                // std::cerr << "Size of continuty points: " << continuty_points.size() << "\n";

                // std::cerr << "Continuity point for face " << face_counter << ": "
                //           << continuty_points[face_counter][0] << ", "
                //           << continuty_points[face_counter][1] << ", "
                //           << continuty_points[face_counter][2] << "\n";

                continuty_points[face_counter] = loc_face_centers[local_face_index];
                ++face_counter;
            }
            // std::cerr << "Face data gathered for cell " << cell_ind << "\n";
            basis_functions = basis_constructor.compute_basis_functions(continuty_points);
            std::cerr << "Basis functions computed\n";

            for (const int face_ind : interaction_region.faces_of_cells().at(cell_ind))
            {
                const int local_face_index = interaction_region.faces().at(face_ind);
                std::vector<double> flux_expr = nK(loc_face_normals[local_face_index], tensor,
                                                   cell_ind, num_nodes_of_face.at(face_ind));

                // Here we need a map to the local flux index to get the right storage in the
                // matrices.
                const int sign = grid.sign_of_face_cell(face_ind, cell_ind);
                std::cerr << "nK computed for face " << local_face_index << " with sign " << sign
                          << "\n";

                std::vector<double> vals = nKgrad(flux_expr, basis_functions);

                std::cerr << "nKgrad computed, setting matrix values\n";

                balance_cells(local_face_index, loc_cell_ind) = sign * vals[0];

                std::cerr << "Value for balance_cells at (" << local_face_index << ", "
                          << loc_cell_ind << ") set to " << sign * vals[0] << "\n";

                for (int i = 1; i < DIM + 1; ++i)
                {
                    const int face_index_secondary =
                        interaction_region.faces_of_cells().at(cell_ind)[i - 1];
                    const int face_index_secondary_local =
                        interaction_region.faces().at(face_index_secondary);

                    balance_faces(local_face_index, face_index_secondary_local) -= sign * vals[i];
                }

                if (cell_ind == interaction_region.main_cell_of_faces().at(local_face_index))
                {
                    // If this is the main cell for the face, we store the flux in the
                    // balance_faces matrix.
                    flux_cells(local_face_index, loc_cell_ind) = sign * vals[0];
                    for (int i = 1; i < DIM + 1; ++i)
                    {
                        const int face_index_secondary =
                            interaction_region.faces_of_cells().at(cell_ind)[i - 1];
                        const int face_index_secondary_local =
                            interaction_region.faces().at(face_index_secondary);

                        flux_faces(local_face_index, face_index_secondary_local) = sign * vals[i];
                    }
                }
            }
        }  // End iteration of cells of the interaction region.

        std::cerr << "Balance matrices computed for interaction region\n";
        std::cerr << "Balance cells matrix:\n" << balance_cells << "\n";
        std::cerr << "Balance faces matrix:\n" << balance_faces << "\n";

        std::cerr << "Flux matrix for cells:\n" << flux_cells << "\n";
        std::cerr << "Flux matrix for faces:\n" << flux_faces << "\n";

        // Compute the inverse of balance_faces matrix.
        MatrixXd balance_faces_inv = balance_faces.inverse();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flux;
        flux = flux_faces * balance_faces_inv * balance_cells + flux_cells;

        std::cerr << "Flux matrix computed for interaction region\n";
        std::cerr << "Flux matrix:\n" << flux << "\n";

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
    }  // End iteration of nodes in the grid.
    // Create the global flux storage. First find the indices that can be used to sort
    // flux_matrix_row_idx
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
            // If we have a new row, we need to update the row_ptr.
            row_ptr.push_back(flux_values.size());
            previous_row = flux_matrix_row_idx[index];
        }

        int counter = 0;
        for (const double col_index : flux_matrix_col_idx[index])
        {
            // Store the flux values in a map to avoid duplicates.
            flux_matrix_values_map[col_index] += flux_matrix_values[index][counter];
            ++counter;
        }
    }
    for (const auto& pair : flux_matrix_values_map)
    {
        col_idx.push_back(pair.first);
        flux_values.push_back(pair.second);
    }
    // Add the last row pointer.
    row_ptr.push_back(col_idx.size());

    // Create the compressed data storage for the flux.
    auto flux_storage = std::make_shared<CompressedDataStorage<double>>(
        grid.num_faces(), grid.num_cells(), row_ptr, col_idx, flux_values);
    // Create the scalar discretization object and return it.
    ScalarDiscretization discretization;
    discretization.flux = flux_storage;
    return discretization;
}
