#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <map>
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
    for (const int face_ind : interaction_region.faces())
    {
        centers.push_back(grid.face_center(face_ind));
    }
    return centers;
}

// Helper function to get face normals for all faces in an interaction region
std::vector<std::vector<double>> face_normals_of_interaction_region(
    const InteractionRegion& interaction_region, const Grid& grid)
{
    std::vector<std::vector<double>> normals;
    for (const int face_ind : interaction_region.faces())
    {
        normals.push_back(grid.face_normal(face_ind));
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
    std::vector<std::vector<double>> continuty_points;

    std::vector<std::vector<double>> basis_functions;

    BasisConstructor basis_constructor(grid.dim());

    const int DIM = grid.dim();
    // Preallocate the continuity points based on the grid dimension.
    if (DIM == 2)
    {
        continuty_points = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    }
    else if (DIM == 3)
    {
        continuty_points = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    }
    else
    {
        throw std::runtime_error("Unsupported grid dimension");
    }

    for (int node_ind{0}; node_ind < grid.num_nodes(); ++node_ind)
    {
        // Get the interaction region for the node.
        InteractionRegion interaction_region(node_ind, 1, &grid);

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

            int face_counter = 0;
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
        }  // Cells of the interaction region. Next step is to invert the face balance matrix and
           // get the fluxes.
    }
}
