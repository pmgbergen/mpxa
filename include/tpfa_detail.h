#ifndef TPFA_DETAIL_H
#define TPFA_DETAIL_H

// Internal helpers for the TPFA discretisation, exposed in a named namespace so
// that unit tests can access them directly without going through the public tpfa()
// entry point.

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "discr.h"
#include "grid.h"
#include "stencil_data.h"
#include "tensor.h"

namespace tpfa_detail
{

// All coordinate and vector-source work is done in 3D space.
constexpr int SPATIAL_DIM = 3;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// Side-specific geometry and transmissibility for one cell adjacent to a face.
struct FaceSideData
{
    int face_ind;
    int cell_ind;
    int sign;
    double trm;
    std::array<double, SPATIAL_DIM> face_cell_vec;
};

// Bundles the stencil accumulators for a tpfa() call.
struct TpfaAccumulator
{
    FluxStencilData& flux;
    BoundaryStencilData& boundary;
};

// ---------------------------------------------------------------------------
// Transmissibility helper
// ---------------------------------------------------------------------------

// Compute n·K·(face_center − cell_center) / |face_center − cell_center|² for
// one cell side.  sign is +1 or −1 (from sign_of_face_cell).
double nKproj(const std::vector<double>& face_normal, const SecondOrderTensor& tensor,
              const std::array<double, SPATIAL_DIM>& cell_face_vec, int sign, int cell_ind);

// ---------------------------------------------------------------------------
// Geometry / initialisation helpers
// ---------------------------------------------------------------------------

// Compute the FaceSideData for cell cell_ind adjacent to face face_ind.
FaceSideData compute_face_side_data(int face_ind, int cell_ind, const Grid& grid,
                                    const SecondOrderTensor& tensor);

// Initialise a FluxStencilData with capacity reserved for num_faces rows.
FluxStencilData init_tpfa_flux_stencil(int num_faces);

// Initialise a BoundaryStencilData with capacity reserved for num_boundary_faces rows.
BoundaryStencilData init_tpfa_boundary_stencil(int num_boundary_faces);

// ---------------------------------------------------------------------------
// Boundary sub-helpers
// ---------------------------------------------------------------------------

// Add the Dirichlet contribution to the flux stencil for one boundary face.
void add_dirichlet_flux_entry(FluxStencilData& flux, const FaceSideData& side);

// Add the Dirichlet contribution to the boundary stencil for one boundary face.
void add_dirichlet_boundary_entries(BoundaryStencilData& boundary, const FaceSideData& side);

// Add the Neumann contribution to the boundary stencil for one boundary face.
void add_neumann_boundary_entries(BoundaryStencilData& boundary, const FaceSideData& side);

// ---------------------------------------------------------------------------
// Face accumulation
// ---------------------------------------------------------------------------

// Accumulate transmissibility and vector-source contributions for an internal face.
void accumulate_internal_face(TpfaAccumulator& acc, const FaceSideData& side_a,
                               const FaceSideData& side_b);

// Accumulate contributions for a boundary face, dispatching on bc type.
void accumulate_boundary_face(TpfaAccumulator& acc, const FaceSideData& side,
                               BoundaryCondition bc);

}  // namespace tpfa_detail

#endif  // TPFA_DETAIL_H
