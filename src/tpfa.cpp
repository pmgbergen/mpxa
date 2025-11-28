#include <array>
#include <iostream>
#include <map>
#include <vector>

#include "../include/discr.h"

namespace  // Anonymous namespace for helper functions.
{
// Helper function to compute the product between normal vector, tensor, and cell-face
// vector.
const double nKproj(const std::vector<double>& face_normal, const SecondOrderTensor& tensor,
                    const std::array<double, 3>& cell_face_vec, const int sign, const int cell_ind)
{
    // Compute the squared distance between the cell center and the face center. We get
    // one power of the distance to make the cell-face vector a unit vector, and a
    // second power to get a distance measure (a gradient).
    double dist = 0.0;
    const int dim = face_normal.size();

    for (int i{0}; i < dim; ++i)
    {
        dist += cell_face_vec[i] * cell_face_vec[i];
    }

    if (tensor.is_isotropic())
    {
        double proj = 0.0;

        for (int i{0}; i < dim; ++i)
        {
            proj += sign * face_normal[i] * cell_face_vec[i];
        }
        return tensor.isotropic_data(cell_ind) * proj / dist;
    }
    else if (tensor.is_diagonal())
    {
        double prod = 0.0;
        auto diag = tensor.diagonal_data(cell_ind);
        for (int i{0}; i < dim; ++i)
        {
            prod += sign * face_normal[i] * cell_face_vec[i] * diag[i];
        }
        return prod / dist;
    }
    else
    {
        double prod = 0.0;
        auto full_data = tensor.full_data(cell_ind);
        for (int i{0}; i < dim; ++i)
        {
            for (int j{0}; j < dim; ++j)
            {
                if (i == j) {
                    double tensor_val = full_data[i];  // 0,1,2 for diagonal
                    prod += sign * face_normal[i] * cell_face_vec[j] * tensor_val;
                } else {
                    int k = 2 + i + j;
                    double tensor_val = full_data[k];
                    prod += sign * face_normal[i] * cell_face_vec[j] * tensor_val;
                }
            }
        }
        return prod / dist;
    }
}
}  // namespace

ScalarDiscretization tpfa(const Grid& grid, const SecondOrderTensor& tensor,
                          const std::map<int, BoundaryCondition>& bc_map)
{
    const int num_boundary_faces = bc_map.size();
    const int num_internal_faces = grid.num_faces() - num_boundary_faces;

    // All coordinates are in 3D space, and the tensor is a 3x3 matrix no matter what.
    constexpr int DIM = 3;

    std::vector<int> row_ptr_flux;
    row_ptr_flux.reserve(grid.num_faces() + 1);
    row_ptr_flux.push_back(0);

    // Reserve space for the transmissibility matrix. The size is 2 * num_internal_faces
    // + num_boundary_faces.
    std::vector<double> trm;
    std::vector<int> col_idx_flux;
    trm.reserve(2 * num_internal_faces + num_boundary_faces);
    col_idx_flux.reserve(2 * num_internal_faces + num_boundary_faces);

    // Boundary flux matrix. There will be an estimated num_boundary_faces entries in
    // the matrix.
    std::vector<double> trm_bound;
    trm_bound.reserve(num_boundary_faces);

    std::vector<int> row_ptr_bound_flux;
    row_ptr_bound_flux.reserve(grid.num_faces() + 1);
    row_ptr_bound_flux.push_back(0);

    std::vector<int> col_idx_bound_flux;
    col_idx_bound_flux.reserve(num_boundary_faces);

    // Discretization for the boundary face pressure reconstruction from cell pressures.
    std::vector<double> bound_pressure_cell;
    std::vector<int> row_ptr_bound_pressure_cell;
    std::vector<int> col_idx_bound_pressure_cell;
    bound_pressure_cell.reserve(num_boundary_faces);
    row_ptr_bound_pressure_cell.reserve(grid.num_cells() + 1);
    row_ptr_bound_pressure_cell.push_back(0);
    col_idx_bound_pressure_cell.reserve(num_boundary_faces);

    // Discretization for the boundary face pressure reconstruction from boundary
    // conditions.
    std::vector<double> bound_pressure_face;
    std::vector<int> row_ptr_bound_pressure_face;
    std::vector<int> col_idx_bound_pressure_face;
    bound_pressure_face.reserve(num_boundary_faces);
    row_ptr_bound_pressure_face.reserve(grid.num_faces() + 1);
    row_ptr_bound_pressure_face.push_back(0);
    col_idx_bound_pressure_face.reserve(num_boundary_faces);

    // Discretization for the vector source.
    std::vector<double> vector_source;
    std::vector<int> row_ptr_vector_source;
    std::vector<int> col_idx_vector_source;
    vector_source.reserve(grid.num_faces() * 3);
    row_ptr_vector_source.reserve(grid.num_faces() + 1);
    col_idx_vector_source.reserve(grid.num_faces() * 3);
    row_ptr_vector_source.push_back(0);

    // Discretization for the vector source contribution to the boundary face pressure
    // reconstruction.
    std::vector<double> vector_source_bound;
    vector_source_bound.reserve(num_boundary_faces);
    std::vector<int> col_idx_vector_source_bound;
    col_idx_vector_source_bound.reserve(num_boundary_faces);
    std::vector<int> row_ptr_vector_source_bound;
    row_ptr_vector_source_bound.reserve(grid.num_faces() + 1);
    row_ptr_vector_source_bound.push_back(0);

    // Preallocate holders of local geometric data.
    std::array<double, DIM> face_cell_a_vec{};
    std::array<double, DIM> face_cell_b_vec{};
    std::vector<double> face_center(DIM);
    std::vector<double> normal(DIM);

    for (int face_ind{0}; face_ind < grid.num_faces(); ++face_ind)
    {
        // Get various properties of the face and its first neighboring cell.
        auto cells = grid.cells_of_face(face_ind);
        face_center = grid.face_center(face_ind);
        normal = grid.face_normal(face_ind);
        const int cell_a = cells[0];
        const int sign_a = grid.sign_of_face_cell(face_ind, cell_a);

        for (int i{0}; i < DIM; ++i)
        {
            face_cell_a_vec[i] = face_center[i] - grid.cell_center(cell_a)[i];
        }
        const double trm_a = nKproj(normal, tensor, face_cell_a_vec, sign_a, cell_a);

        if (cells.size() == 2)  // Internal face.
        {
            // Get the second neighboring cell and its properties.
            const int cell_b = cells[1];
            const int sign_b = grid.sign_of_face_cell(face_ind, cell_b);

            for (int i{0}; i < DIM; ++i)
            {
                face_cell_b_vec[i] = face_center[i] - grid.cell_center(cell_b)[i];
            }

            const double trm_b = nKproj(normal, tensor, face_cell_b_vec, sign_b, cell_b);
            const double harmonic_mean = trm_a * trm_b / (trm_a + trm_b);
            trm.push_back(harmonic_mean * sign_a);
            trm.push_back(harmonic_mean * sign_b);
            col_idx_flux.push_back(cell_a);
            col_idx_flux.push_back(cell_b);

            // Also compute the vector source term for the face, for both cells.
            for (int i{0}; i < DIM; ++i)
            {
                // Compute the vector source term for the face.
                vector_source.push_back(harmonic_mean * sign_a * face_cell_a_vec[i]);
                col_idx_vector_source.push_back(cell_a * DIM + i);
            }
            for (int i{0}; i < DIM; ++i)
            {
                // Compute the vector source term for the face.
                vector_source.push_back(harmonic_mean * sign_b * face_cell_b_vec[i]);
                col_idx_vector_source.push_back(cell_b * DIM + i);
            }

            // Store the flux in the compressed data storage.
            // flux->set_value(face_ind, flux);
        }
        else  // Boundary face.
        {
            const BoundaryCondition bc = bc_map.at(face_ind);

            switch (bc)  // Corrected to use the scoped enum directly.
            {
                case BoundaryCondition::Dirichlet:
                    // The transmissibility for Dirichlet conditions is the same as the
                    // (half) transmissibility for the internal face.
                    trm.push_back(trm_a * sign_a);
                    col_idx_flux.push_back(cell_a);
                    // The transmissibility for the boundary face is the negative of
                    // the transmissibility for the internal face.
                    trm_bound.push_back(-trm_a * sign_a);
                    col_idx_bound_flux.push_back(face_ind);

                    // Boundary face pressure reconstruction. There is no contribution
                    // from the cell, while the face contributes a unit value.
                    bound_pressure_face.push_back(1.0);
                    col_idx_bound_pressure_face.push_back(face_ind);

                    // The vector source term for the Dirichlet condition is half the
                    // calculation for the internal face. There is no contribution to
                    // the boundary discretization for the vector source term.
                    for (int i{0}; i < DIM; ++i)
                    {
                        // Compute the vector source term for the face.
                        vector_source.push_back(trm_a * sign_a * face_cell_a_vec[i]);
                        col_idx_vector_source.push_back(cell_a * DIM + i);
                    }

                    break;

                case BoundaryCondition::Neumann:
                    // Neumann conditions have no transmissibility for the flux matrix.
                    // The bounday flux is set to unity, as this will transmit the
                    // Neumann condition to the cell.
                    trm_bound.push_back(1.0 * sign_a);
                    col_idx_bound_flux.push_back(face_ind);

                    // Boundary face pressure reconstruction. The cell contributes a
                    // unit value.
                    bound_pressure_cell.push_back(1.0);
                    col_idx_bound_pressure_cell.push_back(cell_a);
                    // The face contribution equals to the offset from the cell value
                    // due to the imposed Neumann condition. This is the negative
                    // inverse of the transmissibility.
                    bound_pressure_face.push_back(-1.0 / trm_a);
                    col_idx_bound_pressure_face.push_back(face_ind);

                    // There is no vector source term for the Neumann condition, no need
                    // to add anything.
                    for (int i{0}; i < DIM; ++i)
                    {
                        // Compute the vector source term for the face.
                        vector_source_bound.push_back(face_cell_a_vec[i]);
                        col_idx_vector_source_bound.push_back(cell_a * DIM + i);
                    }

                    break;

                case BoundaryCondition::Robin:
                    // Handle Robin boundary condition.
                    throw std::logic_error("Robin boundary condition not implemented");

                default:
                    throw std::runtime_error("Unknown boundary condition type");
            }
        }
        // Move the row pointers to the next face. Apply to both flux and bound_flux.
        row_ptr_flux.push_back(col_idx_flux.size());
        row_ptr_bound_flux.push_back(col_idx_bound_flux.size());
        row_ptr_vector_source.push_back(vector_source.size());
        row_ptr_vector_source_bound.push_back(vector_source_bound.size());
        row_ptr_bound_pressure_cell.push_back(col_idx_bound_pressure_cell.size());
        row_ptr_bound_pressure_face.push_back(col_idx_bound_pressure_face.size());
    }

    // Create the ScalarDiscretization object and return it.
    ScalarDiscretization discr;
    discr.flux = std::make_shared<CompressedDataStorage<double>>(grid.num_faces(), grid.num_cells(),
                                                                 row_ptr_flux, col_idx_flux, trm);
    discr.bound_flux = std::make_shared<CompressedDataStorage<double>>(
        grid.num_faces(), grid.num_faces(), row_ptr_bound_flux, col_idx_bound_flux, trm_bound);
    discr.vector_source = std::make_shared<CompressedDataStorage<double>>(
        grid.num_faces(), grid.num_cells() * DIM, row_ptr_vector_source, col_idx_vector_source,
        vector_source);
    discr.bound_pressure_vector_source = std::make_shared<CompressedDataStorage<double>>(
        grid.num_faces(), grid.num_cells() * DIM, row_ptr_vector_source_bound,
        col_idx_vector_source_bound, vector_source_bound);
    discr.bound_pressure_cell = std::make_shared<CompressedDataStorage<double>>(
        grid.num_faces(), grid.num_cells(), row_ptr_bound_pressure_cell,
        col_idx_bound_pressure_cell, bound_pressure_cell);
    discr.bound_pressure_face = std::make_shared<CompressedDataStorage<double>>(
        grid.num_faces(), grid.num_faces(), row_ptr_bound_pressure_face,
        col_idx_bound_pressure_face, bound_pressure_face);
    return discr;
}
