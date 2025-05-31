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
                             const SecondOrderTensor& tensor, const int cell_ind)
{
    // Compute the product between the normal vector, the tensor, and the cell-face
    // vector.

    const int dim = face_normal.size();
    std::vector<double> result(dim, 0.0);

    if (tensor.is_isotropic())
    {
        for (int i{0}; i < dim; ++i)
        {
            result[i] = -face_normal[i] * tensor.isotropic_data(cell_ind);
        }
    }
    else if (tensor.is_diagonal())
    {
        std::vector<double> diag = tensor.diagonal_data(cell_ind);
        for (int i{0}; i < dim; ++i)
        {
            result[i] = -face_normal[i] * diag[i];
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
                result[i] -= face_normal[j] * tensor_val;
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
        for (size_t j = 0; j < basis_functions.size(); ++j)
        {
            grad[i] += nK[j] * basis_functions[i][j];
        }
    }
    return grad;
}

}  // namespace

ScalarDiscretization mpfa(const Grid& grid, const SecondOrderTensor& tensor,
                          const std::map<int, BoundaryCondition>& bc_map)
{
    // Data structures for the discretization process.
    std::vector<std::vector<double>> continuty_points;
    std::vector<std::vector<double>> basis_functions;
    BasisConstructor basis_constructor(grid.dim());

    // Data structures for the computed stencils.
    std::vector<std::vector<double>> flux_matrix_values;
    std::vector<int> flux_matrix_row_idx;
    std::vector<std::vector<int>> flux_matrix_col_idx;

    const int DIM = grid.dim();
    int tot_num_transmissibilities = 0;

    for (int node_ind{0}; node_ind < grid.num_nodes(); ++node_ind)
    {
        // Get the interaction region for the node.
        InteractionRegion interaction_region(node_ind, 1, grid);

        const int num_faces = interaction_region.faces().size();
        const int num_cells = interaction_region.cells().size();

        // Initialize matrices for the discretization.
        MatrixXd balance_cells(num_faces, num_cells);
        MatrixXd balance_faces(num_faces, num_faces);

        MatrixXd flux_cells(num_faces, num_cells);
        MatrixXd flux_faces(num_faces, num_faces);

        // TODO: Should we use vectors for the inner quantities?
        std::vector<std::vector<double>> loc_cell_centers =
            cell_centers_of_interaction_region(interaction_region, grid);
        std::vector<std::vector<double>> loc_face_centers =
            face_centers_of_interaction_region(interaction_region, grid);
        std::vector<std::vector<double>> loc_face_normals =
            face_normals_of_interaction_region(interaction_region, grid);

        // Iterate over the faces in the interaction region.
        for (int loc_cell_ind{0}; loc_cell_ind < num_cells; ++loc_cell_ind)
        {
            continuty_points[0] = loc_cell_centers[loc_cell_ind];
            const int cell_ind = interaction_region.cells().at(loc_cell_ind);

            int face_counter = 1;
            for (const int face_ind : interaction_region.faces_of_cells().at(cell_ind))
            {
                // Get the face normal and center.
                const int local_face_index = interaction_region.faces().at(face_ind);

                continuty_points[face_counter] = loc_face_centers[local_face_index];
                ++face_counter;
            }
            basis_functions = basis_constructor.compute_basis_functions(continuty_points);

            for (const int face_ind : interaction_region.faces_of_cells().at(cell_ind))
            {
                const int local_face_index = interaction_region.faces().at(face_ind);
                std::vector<double> flux_expr =
                    nK(loc_face_normals[local_face_index], tensor, cell_ind);

                // Here we need a map to the local flux index to get the right storage in the
                // matrices.
                const int sign = grid.sign_of_face_cell(face_ind, cell_ind);

                std::vector<double> vals = nKgrad(flux_expr, basis_functions);

                balance_cells(local_face_index, cell_ind) = sign * vals[0];

                for (int i = 1; i < DIM + 1; ++i)
                {
                    balance_cells(local_face_index,
                                  interaction_region.faces_of_cells().at(loc_cell_ind)[i - 1]) =
                        sign * vals[i];
                }

                if (cell_ind == interaction_region.main_cell_of_faces().at(local_face_index))
                {
                    // If this is the main cell for the face, we store the flux in the
                    // balance_faces matrix.
                    balance_faces(local_face_index, local_face_index) = sign * vals[0];
                    for (int i = 1; i < DIM + 1; ++i)
                    {
                        balance_cells(local_face_index,
                                      interaction_region.faces_of_cells().at(loc_cell_ind)[i - 1]) =
                            sign * vals[i];
                    }
                }
            }
        }  // End iteration of cells of the interaction region.

        // Compute the inverse of balance_faces matrix.
        MatrixXd balance_faces_inv = balance_faces.inverse();
        MatrixXd flux = flux_faces * balance_faces_inv * balance_cells + flux_cells;
        // Store the computed flux in the flux_matrix_values, row_idx, and col_idx.
        for (int i = 0; i < num_faces; ++i)
        {
            std::vector<double> row(flux.row(i).data(), flux.row(i).data() + flux.cols());
            flux_matrix_values.push_back(row);
            flux_matrix_row_idx.push_back(interaction_region.faces().at(i));
            flux_matrix_col_idx.push_back(interaction_region.cells());
        }
        tot_num_transmissibilities += num_faces * num_cells;
    }  // End iteration of nodes in the grid.
    // Create the global flux storage. First find the indices that can be used to sort
    // flux_matrix_row_idx
    std::vector<int> sorted_indices(flux_matrix_row_idx.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);  // Fill with 0, 1, 2, ...
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&flux_matrix_row_idx](int i1, int i2)
              { return flux_matrix_row_idx[i1] < flux_matrix_row_idx[i2]; });

    std::vector<int> row_ptr(grid.num_faces() + 1, 0);
    std::vector<int> col_idx;
    std::vector<double> flux_values;
    col_idx.reserve(tot_num_transmissibilities);
    flux_values.reserve(tot_num_transmissibilities);

    int previous_row = -1;

    for (const int index : sorted_indices)
    {
        // If the row index has changed, we need to update the row_ptr.
        if (flux_matrix_row_idx[index] != previous_row)
        {
            // If we have a new row, we need to update the row_ptr.
            row_ptr.push_back(col_idx.size());
            previous_row = flux_matrix_row_idx[index];
        }
        // Fill the col_idx and flux_values vectors based on the sorted indices.
        col_idx.insert(col_idx.end(), flux_matrix_col_idx[index].begin(),
                       flux_matrix_col_idx[index].end());
        flux_values.insert(flux_values.end(), flux_matrix_values[index].begin(),
                           flux_matrix_values[index].end());
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
