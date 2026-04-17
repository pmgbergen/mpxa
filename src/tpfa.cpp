#include <array>
#include <unordered_map>
#include <vector>

#include "../include/discr.h"
#include "../include/stencil_data.h"

namespace tpfa_detail  // Helper functions for tpfa().
{
// Helper function to compute the product between normal vector, tensor, and cell-face
// vector.
double nKproj(const std::vector<double>& face_normal, const SecondOrderTensor& tensor,
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
    if (std::abs(dist) < 1e-20) {
        // The threshold is arbitrary small.
        throw std::runtime_error("Division by zero in nKproj");
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
                    // Full tensor storage: [K_00, K_11, K_22, K_01, K_02, K_12].
                    // Off-diagonal pairs (i,j) with i≠j map to indices 3,4,5 via k = 2+i+j
                    // because i+j gives 1,2,3 for (0,1),(0,2),(1,2) and their symmetric pairs.
                    const int k = 2 + i + j;
                    double tensor_val = full_data[k];
                    prod += sign * face_normal[i] * cell_face_vec[j] * tensor_val;
                }
            }
        }
        return prod / dist;
    }
}

// Bundles the stencil accumulators for a tpfa() call.
struct TpfaAccumulator
{
    FluxStencilData& flux;
    BoundaryStencilData& boundary;
};

// Accumulates transmissibility and vector-source contributions for an internal face.
void accumulate_internal_face(TpfaAccumulator& acc, const int face_ind, const int cell_a,
                               const int cell_b, const int sign_a, const int sign_b,
                               const double trm_a, const double trm_b,
                               const std::array<double, 3>& face_cell_a_vec,
                               const std::array<double, 3>& face_cell_b_vec, const int dim)
{
    const double harmonic_mean = trm_a * trm_b / (trm_a + trm_b);

    acc.flux.row_idx.push_back(face_ind);
    acc.flux.col_idx.push_back({cell_a, cell_b});
    acc.flux.flux_values.push_back({harmonic_mean * sign_a, harmonic_mean * sign_b});

    std::vector<double> vs_row;
    vs_row.reserve(2 * dim);
    for (int i = 0; i < dim; ++i)
        vs_row.push_back(harmonic_mean * sign_a * face_cell_a_vec[i]);
    for (int i = 0; i < dim; ++i)
        vs_row.push_back(harmonic_mean * sign_b * face_cell_b_vec[i]);
    acc.flux.vs_values.push_back(std::move(vs_row));
}

// Accumulates transmissibility contributions for a boundary face.
void accumulate_boundary_face(TpfaAccumulator& acc, const int face_ind, const int cell_a,
                               const int sign_a, const double trm_a,
                               const std::array<double, 3>& face_cell_a_vec, const int dim,
                               const BoundaryCondition bc)
{
    switch (bc)
    {
        case BoundaryCondition::Dirichlet:
            acc.flux.row_idx.push_back(face_ind);
            acc.flux.col_idx.push_back({cell_a});
            acc.flux.flux_values.push_back({trm_a * sign_a});
            {
                std::vector<double> vs_row;
                vs_row.reserve(dim);
                for (int i = 0; i < dim; ++i)
                    vs_row.push_back(trm_a * sign_a * face_cell_a_vec[i]);
                acc.flux.vs_values.push_back(std::move(vs_row));
            }
            acc.boundary.bound_flux.row_idx.push_back(face_ind);
            acc.boundary.bound_flux.col_idx.push_back({face_ind});
            acc.boundary.bound_flux.values.push_back({-trm_a * sign_a});
            acc.boundary.pressure_face.row_idx.push_back(face_ind);
            acc.boundary.pressure_face.col_idx.push_back({face_ind});
            acc.boundary.pressure_face.values.push_back({1.0});
            break;

        case BoundaryCondition::Neumann:
            // Neumann faces have no cell contribution to the flux matrix (empty row).
            acc.boundary.bound_flux.row_idx.push_back(face_ind);
            acc.boundary.bound_flux.col_idx.push_back({face_ind});
            acc.boundary.bound_flux.values.push_back({static_cast<double>(sign_a)});
            acc.boundary.pressure_cell.row_idx.push_back(face_ind);
            acc.boundary.pressure_cell.col_idx.push_back({cell_a});
            acc.boundary.pressure_cell.values.push_back({1.0});
            acc.boundary.pressure_face.row_idx.push_back(face_ind);
            acc.boundary.pressure_face.col_idx.push_back({face_ind});
            acc.boundary.pressure_face.values.push_back({-1.0 / trm_a});
            {
                std::vector<double> vs_row;
                std::vector<int> vs_col;
                vs_row.reserve(dim);
                vs_col.reserve(dim);
                for (int i = 0; i < dim; ++i)
                {
                    vs_row.push_back(face_cell_a_vec[i]);
                    vs_col.push_back(cell_a * dim + i);
                }
                acc.boundary.vector_source.row_idx.push_back(face_ind);
                acc.boundary.vector_source.col_idx.push_back(std::move(vs_col));
                acc.boundary.vector_source.values.push_back(std::move(vs_row));
            }
            break;

        case BoundaryCondition::Robin:
            throw std::logic_error("Robin boundary condition not implemented");

        default:
            throw std::runtime_error("Unknown boundary condition type");
    }
}

}  // namespace tpfa_detail

ScalarDiscretization tpfa(const Grid& grid, const SecondOrderTensor& tensor,
                          const std::unordered_map<int, BoundaryCondition>& bc_map)
{
    using namespace tpfa_detail;

    const int num_boundary_faces = static_cast<int>(bc_map.size());
    const int num_internal_faces = grid.num_faces() - num_boundary_faces;

    // All coordinates are in 3D space, and the tensor is a 3×3 matrix no matter what.
    constexpr int DIM = 3;

    FluxStencilData flux_stencil;
    flux_stencil.row_idx.reserve(grid.num_faces());
    flux_stencil.col_idx.reserve(grid.num_faces());
    flux_stencil.flux_values.reserve(grid.num_faces());
    flux_stencil.vs_values.reserve(grid.num_faces());

    BoundaryStencilData bound_stencil;
    bound_stencil.bound_flux.reserve(num_boundary_faces);
    bound_stencil.pressure_cell.reserve(num_boundary_faces);
    bound_stencil.pressure_face.reserve(num_boundary_faces);
    bound_stencil.vector_source.reserve(num_boundary_faces);

    TpfaAccumulator acc{flux_stencil, bound_stencil};

    // Preallocate holders of local geometric data (reused each iteration).
    std::array<double, DIM> face_cell_a_vec{};
    std::array<double, DIM> face_cell_b_vec{};

    for (int face_ind{0}; face_ind < grid.num_faces(); ++face_ind)
    {
        const auto cells = grid.cells_of_face(face_ind);
        const auto& face_center = grid.face_center(face_ind);
        const auto& normal = grid.face_normal(face_ind);
        const int cell_a = cells[0];
        const int sign_a = grid.sign_of_face_cell(face_ind, cell_a);

        for (int i{0}; i < DIM; ++i)
            face_cell_a_vec[i] = face_center[i] - grid.cell_center(cell_a)[i];

        const double trm_a = nKproj(normal, tensor, face_cell_a_vec, sign_a, cell_a);

        if (cells.size() == 2)  // Internal face.
        {
            const int cell_b = cells[1];
            const int sign_b = grid.sign_of_face_cell(face_ind, cell_b);

            for (int i{0}; i < DIM; ++i)
                face_cell_b_vec[i] = face_center[i] - grid.cell_center(cell_b)[i];

            const double trm_b = nKproj(normal, tensor, face_cell_b_vec, sign_b, cell_b);

            accumulate_internal_face(acc, face_ind, cell_a, cell_b, sign_a, sign_b, trm_a,
                                     trm_b, face_cell_a_vec, face_cell_b_vec, DIM);
        }
        else  // Boundary face.
        {
            accumulate_boundary_face(acc, face_ind, cell_a, sign_a, trm_a, face_cell_a_vec,
                                     DIM, bc_map.at(face_ind));
        }
    }

    // Convert stencils to CSR matrices and assemble the discretization.
    ScalarDiscretization discr;

    auto [flux_mat, vs_mat] =
        flux_stencil_to_csr(flux_stencil, grid.num_faces(), grid.num_cells(), DIM);
    discr.flux = flux_mat;
    discr.vector_source = vs_mat;

    discr.bound_flux = stencil_to_csr(bound_stencil.bound_flux, grid.num_faces(),
                                       grid.num_faces());
    discr.bound_pressure_cell = stencil_to_csr(bound_stencil.pressure_cell,
                                                grid.num_faces(), grid.num_cells());
    discr.bound_pressure_face = stencil_to_csr(bound_stencil.pressure_face,
                                                grid.num_faces(), grid.num_faces());
    discr.bound_pressure_vector_source =
        stencil_to_csr(bound_stencil.vector_source, grid.num_faces(),
                        grid.num_cells() * DIM);

    return discr;
}
