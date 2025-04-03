#include <vector>

#include "discr.h"

namespace  // Anonymous namespace for helper functions.
{
// Helper function to compute the product between normal vector, tensor, and cell-face
// vector.
const double nKproj(const double* face_normal, const SecondOrderTensor& tensor,
                    const double* cell_face_vec, const int dim, const int cell_ind)
{
    if (tensor.is_isotropic())
    {
        double proj = 0.0;
        for (int i{0}; i < dim; ++i)
        {
            proj += face_normal[i] * cell_face_vec[i];
        }
        return tensor.isotropic_data()[cell_ind] * proj;
    }
    else if (tensor.is_diagonal())
    {
        double prod = 0.0;
        for (int i{0}; i < dim; ++i)
        {
            prod += face_normal[i] * cell_face_vec[i] * tensor.diagonal_data()[i][cell_ind];
        }
        return prod;
    }
    else
    {
        double prod = 0.0;
        for (int i{0}; i < dim; ++i)
        {
            for (int j{0}; j < dim; ++j)
            {
                double tensor_val;
                if (i == 0 && j == 0)
                    tensor_val = tensor.full_data()[cell_ind][0];
                else if (i == 1 && j == 1)
                    tensor_val = tensor.full_data()[cell_ind][1];
                else if (i == 0 && j == 1 || i == 1 && j == 0)
                    tensor_val = tensor.full_data()[cell_ind][2];
                else if (i == 2 && j == 2)
                    tensor_val = tensor.full_data()[cell_ind][3];
                else if (i == 0 && j == 2 || i == 2 && j == 0)
                    tensor_val = tensor.full_data()[cell_ind][4];
                else if (i == 1 && j == 2 || i == 2 && j == 1)
                    tensor_val = tensor.full_data()[cell_ind][5];

                prod += face_normal[i] * cell_face_vec[j] * tensor_val;
            }
        }
        return prod;
    }
}
}  // namespace

ScalarDiscretization tpfa(const Grid& grid, const SecondOrderTensor& tensor)
{
    std::vector<double> trm;
    trm.reserve(grid.num_faces());  // Should be 2 * num_internal_faces + num_boundary_faces.

    std::vector<int> row_ptr_flux;
    row_ptr_flux.reserve(grid.num_faces() + 1);
    row_ptr_flux.push_back(0);

    std::vector<int> col_idx_flux;
    col_idx_flux.reserve(2 * grid.num_faces());

    std::vector<double> trm_bound;
    trm_bound.reserve(grid.num_faces());  // Should be 2 * num_internal_faces + num_boundary_faces.

    std::vector<int> row_ptr_bound_flux;
    row_ptr_bound_flux.reserve(grid.num_faces() + 1);
    row_ptr_bound_flux.push_back(0);

    std::vector<int> col_idx_bound_flux;
    col_idx_bound_flux.reserve(2 * grid.num_faces());

    for (int face_ind{0}; face_ind < grid.num_faces(); ++face_ind)
    {
        const std::vector<int> cells = grid.cells_of_face(face_ind);
        const int cell_a = cells[0];
        const int sign_a = grid.sign_of_face_cell(face_ind, cell_a);
        const double* normal = grid.face_normal(face_ind);
        const double* center = grid.face_center(face_ind);
        double* face_cell_a_vec = new double[grid.dim()];

        for (int i{0}; i < grid.dim(); ++i)
        {
            face_cell_a_vec[i] = center[i] - grid.cell_center(cell_a)[i];
        }
        const double trm_a = nKproj(normal, tensor, face_cell_a_vec, grid.dim(), cell_a);

        if (cells.size() == 2)  // Internal face.
        {
            const int cell_b = cells[1];
            const int sign_b = grid.sign_of_face_cell(face_ind, cell_b);

            double* face_cell_b_vec = new double[grid.dim()];
            for (int i{0}; i < grid.dim(); ++i)
            {
                face_cell_b_vec[i] = center[i] - grid.cell_center(cell_b)[i];
            }

            const double trm_b = nKproj(normal, tensor, face_cell_b_vec, grid.dim(), cell_b);

            const double harmonic_mean = 1.0 / (1.0 / trm_a + 1.0 / trm_b);

            trm.push_back(harmonic_mean * sign_a);
            trm.push_back(harmonic_mean * sign_b);
            col_idx_flux.push_back(cell_a);
            col_idx_flux.push_back(cell_b);

            // Store the flux in the compressed data storage.
            // flux->set_value(face_ind, flux);

            delete[] face_cell_b_vec;
        }
        else  // Boundary face.
        {
            trm.push_back(trm_a * sign_a);
            col_idx_flux.push_back(cell_a);

            // Store the flux in the compressed data storage.
            // bound_flux->set_value(face_ind, flux);
        }
        row_ptr_flux.push_back(col_idx_flux.size());

        delete[] face_cell_a_vec;
    }

    CompressedDataStorage<double>* flux = new CompressedDataStorage<double>(
        grid.num_faces(), grid.num_cells(), row_ptr_flux, col_idx_flux, trm);

    CompressedDataStorage<double>* bound_flux = new CompressedDataStorage<double>(
        grid.num_faces(), grid.num_cells(), row_ptr_bound_flux, col_idx_bound_flux, trm_bound);

    ScalarDiscretization discr;
    discr.flux = std::unique_ptr<CompressedDataStorage<double>>(flux);
    discr.bound_flux = std::unique_ptr<CompressedDataStorage<double>>(bound_flux);
    return discr;
}
