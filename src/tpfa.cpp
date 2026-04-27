#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../include/discr.h"
#include "../include/stencil_data.h"
#include "../include/tpfa_detail.h"

namespace tpfa_detail
{

double nKproj(const std::vector<double>& face_normal, const SecondOrderTensor& tensor,
              const std::array<double, SPATIAL_DIM>& cell_face_vec, int sign, int cell_ind)
{
    // Squared distance between cell center and face center; provides normalisation
    // for the cell-face unit vector and a distance measure for the gradient.
    double dist = 0.0;
    const int dim = static_cast<int>(face_normal.size());

    for (int i{0}; i < dim; ++i)
        dist += cell_face_vec[i] * cell_face_vec[i];

    if (std::abs(dist) < 1e-20)
        throw std::runtime_error("Division by zero in nKproj");

    if (tensor.is_isotropic())
    {
        double proj = 0.0;
        for (int i{0}; i < dim; ++i)
            proj += sign * face_normal[i] * cell_face_vec[i];
        return tensor.isotropic_data(cell_ind) * proj / dist;
    }
    else if (tensor.is_diagonal())
    {
        double prod = 0.0;
        auto diag = tensor.diagonal_data(cell_ind);
        for (int i{0}; i < dim; ++i)
            prod += sign * face_normal[i] * cell_face_vec[i] * diag[i];
        return prod / dist;
    }
    else
    {
        // Full tensor storage: [K_00, K_11, K_22, K_01, K_02, K_12].
        // Off-diagonal pairs (i,j) with i≠j map to indices 3,4,5 via k = 2+i+j.
        double prod = 0.0;
        auto full_data = tensor.full_data(cell_ind);
        for (int i{0}; i < dim; ++i)
        {
            for (int j{0}; j < dim; ++j)
            {
                const double tensor_val = (i == j) ? full_data[i] : full_data[2 + i + j];
                prod += sign * face_normal[i] * cell_face_vec[j] * tensor_val;
            }
        }
        return prod / dist;
    }
}

FaceSideData compute_face_side_data(int face_ind, int cell_ind, const Grid& grid,
                                    const SecondOrderTensor& tensor)
{
    FaceSideData side;
    side.face_ind = face_ind;
    side.cell_ind = cell_ind;
    side.sign = grid.sign_of_face_cell(face_ind, cell_ind);

    const auto& face_center = grid.face_center(face_ind);
    const auto& cell_center = grid.cell_center(cell_ind);
    side.face_cell_vec = {0.0, 0.0, 0.0};  // zero-pad to SPATIAL_DIM
    const int coord_dim = static_cast<int>(face_center.size());
    for (int i{0}; i < coord_dim; ++i)
        side.face_cell_vec[i] = face_center[i] - cell_center[i];

    side.trm = nKproj(grid.face_normal(face_ind), tensor, side.face_cell_vec, side.sign,
                      cell_ind);
    return side;
}

FluxStencilData init_tpfa_flux_stencil(int num_faces)
{
    FluxStencilData s;
    s.row_idx.reserve(num_faces);
    s.col_idx.reserve(num_faces);
    s.flux_values.reserve(num_faces);
    s.vs_values.reserve(num_faces);
    return s;
}

BoundaryStencilData init_tpfa_boundary_stencil(int num_boundary_faces)
{
    BoundaryStencilData s;
    s.bound_flux.reserve(num_boundary_faces);
    s.pressure_cell.reserve(num_boundary_faces);
    s.pressure_face.reserve(num_boundary_faces);
    s.vector_source.reserve(num_boundary_faces);
    return s;
}

void add_dirichlet_flux_entry(FluxStencilData& flux, const FaceSideData& side)
{
    flux.row_idx.push_back(side.face_ind);
    flux.col_idx.push_back({side.cell_ind});
    flux.flux_values.push_back({side.trm * side.sign});

    std::vector<double> vs_row;
    vs_row.reserve(SPATIAL_DIM);
    for (int i{0}; i < SPATIAL_DIM; ++i)
        vs_row.push_back(side.trm * side.sign * side.face_cell_vec[i]);
    flux.vs_values.push_back(std::move(vs_row));
}

void add_dirichlet_boundary_entries(BoundaryStencilData& boundary, const FaceSideData& side)
{
    boundary.bound_flux.row_idx.push_back(side.face_ind);
    boundary.bound_flux.col_idx.push_back({side.face_ind});
    boundary.bound_flux.values.push_back({-side.trm * side.sign});

    boundary.pressure_face.row_idx.push_back(side.face_ind);
    boundary.pressure_face.col_idx.push_back({side.face_ind});
    boundary.pressure_face.values.push_back({1.0});
}

void add_neumann_boundary_entries(BoundaryStencilData& boundary, const FaceSideData& side)
{
    boundary.bound_flux.row_idx.push_back(side.face_ind);
    boundary.bound_flux.col_idx.push_back({side.face_ind});
    boundary.bound_flux.values.push_back({static_cast<double>(side.sign)});

    boundary.pressure_cell.row_idx.push_back(side.face_ind);
    boundary.pressure_cell.col_idx.push_back({side.cell_ind});
    boundary.pressure_cell.values.push_back({1.0});

    boundary.pressure_face.row_idx.push_back(side.face_ind);
    boundary.pressure_face.col_idx.push_back({side.face_ind});
    boundary.pressure_face.values.push_back({-1.0 / side.trm});

    std::vector<double> vs_row;
    std::vector<int> vs_col;
    vs_row.reserve(SPATIAL_DIM);
    vs_col.reserve(SPATIAL_DIM);
    for (int i{0}; i < SPATIAL_DIM; ++i)
    {
        vs_row.push_back(side.face_cell_vec[i]);
        vs_col.push_back(side.cell_ind * SPATIAL_DIM + i);
    }
    boundary.vector_source.row_idx.push_back(side.face_ind);
    boundary.vector_source.col_idx.push_back(std::move(vs_col));
    boundary.vector_source.values.push_back(std::move(vs_row));
}

void accumulate_internal_face(TpfaAccumulator& acc, const FaceSideData& side_a,
                               const FaceSideData& side_b)
{
    const double harmonic_mean = side_a.trm * side_b.trm / (side_a.trm + side_b.trm);

    acc.flux.row_idx.push_back(side_a.face_ind);
    acc.flux.col_idx.push_back({side_a.cell_ind, side_b.cell_ind});
    acc.flux.flux_values.push_back({harmonic_mean * side_a.sign, harmonic_mean * side_b.sign});

    std::vector<double> vs_row;
    vs_row.reserve(2 * SPATIAL_DIM);
    for (int i{0}; i < SPATIAL_DIM; ++i)
        vs_row.push_back(harmonic_mean * side_a.sign * side_a.face_cell_vec[i]);
    for (int i{0}; i < SPATIAL_DIM; ++i)
        vs_row.push_back(harmonic_mean * side_b.sign * side_b.face_cell_vec[i]);
    acc.flux.vs_values.push_back(std::move(vs_row));
}

void accumulate_boundary_face(TpfaAccumulator& acc, const FaceSideData& side,
                               BoundaryCondition bc)
{
    switch (bc)
    {
        case BoundaryCondition::Dirichlet:
            add_dirichlet_flux_entry(acc.flux, side);
            add_dirichlet_boundary_entries(acc.boundary, side);
            break;

        case BoundaryCondition::Neumann:
            add_neumann_boundary_entries(acc.boundary, side);
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

    FluxStencilData flux_stencil = init_tpfa_flux_stencil(grid.num_faces());
    BoundaryStencilData bound_stencil = init_tpfa_boundary_stencil(num_boundary_faces);
    TpfaAccumulator acc{flux_stencil, bound_stencil};

    for (int face_ind{0}; face_ind < grid.num_faces(); ++face_ind)
    {
        const auto cells = grid.cells_of_face(face_ind);
        const FaceSideData side_a = compute_face_side_data(face_ind, cells[0], grid, tensor);

        if (cells.size() == 2)  // Internal face.
        {
            const FaceSideData side_b = compute_face_side_data(face_ind, cells[1], grid, tensor);
            accumulate_internal_face(acc, side_a, side_b);
        }
        else  // Boundary face.
        {
            accumulate_boundary_face(acc, side_a, bc_map.at(face_ind));
        }
    }

    ScalarDiscretization discr;

    auto [flux_mat, vs_mat] =
        flux_stencil_to_csr(flux_stencil, grid.num_faces(), grid.num_cells(), SPATIAL_DIM);
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
                        grid.num_cells() * SPATIAL_DIM);

    return discr;
}
