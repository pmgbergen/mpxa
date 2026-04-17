#include <gtest/gtest.h>

#include "../../include/discr.h"
#include "../../include/grid.h"
#include "../../include/tensor.h"
#include "../../include/tpfa_detail.h"

using namespace tpfa_detail;

// ============================================================================
// Unit tests for tpfa_detail::nKproj
// ============================================================================

TEST(NKproj, ZeroCellFaceVecThrows)
{
    SecondOrderTensor k(2, 1, {1.0});
    std::array<double, SPATIAL_DIM> zero_vec{0.0, 0.0, 0.0};
    std::vector<double> normal{1.0, 0.0};
    EXPECT_THROW(nKproj(normal, k, zero_vec, 1, 0), std::runtime_error);
}

TEST(NKproj, IsotropicTensor)
{
    // face_normal=[1,0], cell_face_vec=[-0.5,0,0], sign=-1, k=2
    // dist=0.25, proj=(-1)*1*(-0.5)=0.5, trm=2*0.5/0.25=4.0
    SecondOrderTensor k(2, 1, {2.0});
    std::array<double, SPATIAL_DIM> vec{-0.5, 0.0, 0.0};
    std::vector<double> normal{1.0, 0.0};
    EXPECT_NEAR(nKproj(normal, k, vec, -1, 0), 4.0, 1e-10);
}

TEST(NKproj, IsotropicTensorSignFlip)
{
    // Same geometry but sign=+1: trm = 2 * (-0.5) / 0.25 = -4.0
    SecondOrderTensor k(2, 1, {2.0});
    std::array<double, SPATIAL_DIM> vec{-0.5, 0.0, 0.0};
    std::vector<double> normal{1.0, 0.0};
    EXPECT_NEAR(nKproj(normal, k, vec, 1, 0), -4.0, 1e-10);
}

TEST(NKproj, DiagonalTensor)
{
    // face_normal=[1,0], cell_face_vec=[-0.5,0,0], sign=-1, k_xx=2, k_yy=3
    // prod = (-1)*1*(-0.5)*2 + (-1)*0*0*3 = 1.0, trm = 1.0/0.25 = 4.0
    SecondOrderTensor k = SecondOrderTensor(2, 1, {2.0}).with_kyy({3.0});
    std::array<double, SPATIAL_DIM> vec{-0.5, 0.0, 0.0};
    std::vector<double> normal{1.0, 0.0};
    EXPECT_NEAR(nKproj(normal, k, vec, -1, 0), 4.0, 1e-10);
}

TEST(NKproj, DiagonalTensorYDirection)
{
    // face_normal=[0,1], cell_face_vec=[0,-0.5,0], sign=-1, k_xx=2, k_yy=3
    // prod = (-1)*0*0*2 + (-1)*1*(-0.5)*3 = 1.5, trm = 1.5/0.25 = 6.0
    SecondOrderTensor k = SecondOrderTensor(2, 1, {2.0}).with_kyy({3.0});
    std::array<double, SPATIAL_DIM> vec{0.0, -0.5, 0.0};
    std::vector<double> normal{0.0, 1.0};
    EXPECT_NEAR(nKproj(normal, k, vec, -1, 0), 6.0, 1e-10);
}

TEST(NKproj, FullTensor)
{
    // face_normal=[1,0], cell_face_vec=[-0.5,0,0], sign=-1
    // K = [[2,1],[1,3]], full_data=[2,3,0,1,0,0]
    // prod = (-1)*(n[0]*(K_00*v[0]+K_01*v[1]) + n[1]*(K_10*v[0]+K_11*v[1]))
    //      = (-1)*(1*(2*(-0.5)+1*0) + 0*(1*(-0.5)+3*0)) = (-1)*(-1) = 1.0
    // trm = 1.0 / 0.25 = 4.0
    SecondOrderTensor k =
        SecondOrderTensor(2, 1, {2.0}).with_kyy({3.0}).with_kxy({1.0});
    std::array<double, SPATIAL_DIM> vec{-0.5, 0.0, 0.0};
    std::vector<double> normal{1.0, 0.0};
    EXPECT_NEAR(nKproj(normal, k, vec, -1, 0), 4.0, 1e-10);
}

TEST(NKproj, FullTensorOffDiagonalEffect)
{
    // face_normal=[1,0], cell_face_vec=[0,-0.5,0], sign=1
    // K = [[2,1],[1,3]], K_xy=1
    // prod = (1)*(1*(2*0+1*(-0.5)) + 0*(1*0+3*(-0.5))) = 1*(-0.5) = -0.5
    // trm = -0.5 / 0.25 = -2.0
    SecondOrderTensor k =
        SecondOrderTensor(2, 1, {2.0}).with_kyy({3.0}).with_kxy({1.0});
    std::array<double, SPATIAL_DIM> vec{0.0, -0.5, 0.0};
    std::vector<double> normal{1.0, 0.0};
    EXPECT_NEAR(nKproj(normal, k, vec, 1, 0), -2.0, 1e-10);
}

// ============================================================================
// Unit tests for tpfa_detail::init_tpfa_flux_stencil and
//                tpfa_detail::init_tpfa_boundary_stencil
// ============================================================================

TEST(InitStencils, FluxStencilIsEmpty)
{
    auto s = init_tpfa_flux_stencil(10);
    EXPECT_TRUE(s.row_idx.empty());
    EXPECT_TRUE(s.col_idx.empty());
    EXPECT_TRUE(s.flux_values.empty());
    EXPECT_TRUE(s.vs_values.empty());
}

TEST(InitStencils, BoundaryStencilIsEmpty)
{
    auto s = init_tpfa_boundary_stencil(5);
    EXPECT_TRUE(s.bound_flux.row_idx.empty());
    EXPECT_TRUE(s.pressure_cell.row_idx.empty());
    EXPECT_TRUE(s.pressure_face.row_idx.empty());
    EXPECT_TRUE(s.vector_source.row_idx.empty());
}

// ============================================================================
// Unit tests for tpfa_detail::add_dirichlet_flux_entry and
//                tpfa_detail::add_dirichlet_boundary_entries
// ============================================================================

TEST(DirichletHelpers, FluxEntry)
{
    // face=3, cell=0, sign=1, trm=4.0, vec=[-0.5,0,0]
    FaceSideData side;
    side.face_ind = 3;
    side.cell_ind = 0;
    side.sign = 1;
    side.trm = 4.0;
    side.face_cell_vec = {-0.5, 0.0, 0.0};

    FluxStencilData flux;
    add_dirichlet_flux_entry(flux, side);

    ASSERT_EQ(flux.row_idx.size(), 1u);
    EXPECT_EQ(flux.row_idx[0], 3);
    ASSERT_EQ(flux.col_idx[0].size(), 1u);
    EXPECT_EQ(flux.col_idx[0][0], 0);
    EXPECT_NEAR(flux.flux_values[0][0], 4.0, 1e-10);  // trm * sign

    ASSERT_EQ(flux.vs_values[0].size(), 3u);
    EXPECT_NEAR(flux.vs_values[0][0], -2.0, 1e-10);  // 4.0 * 1 * (-0.5)
    EXPECT_NEAR(flux.vs_values[0][1], 0.0, 1e-10);
    EXPECT_NEAR(flux.vs_values[0][2], 0.0, 1e-10);
}

TEST(DirichletHelpers, BoundaryEntries)
{
    // face=3, cell=0, sign=1, trm=4.0
    FaceSideData side;
    side.face_ind = 3;
    side.cell_ind = 0;
    side.sign = 1;
    side.trm = 4.0;
    side.face_cell_vec = {-0.5, 0.0, 0.0};

    BoundaryStencilData boundary;
    add_dirichlet_boundary_entries(boundary, side);

    ASSERT_EQ(boundary.bound_flux.row_idx.size(), 1u);
    EXPECT_EQ(boundary.bound_flux.row_idx[0], 3);
    EXPECT_EQ(boundary.bound_flux.col_idx[0][0], 3);
    EXPECT_NEAR(boundary.bound_flux.values[0][0], -4.0, 1e-10);  // -(trm * sign)

    ASSERT_EQ(boundary.pressure_face.row_idx.size(), 1u);
    EXPECT_EQ(boundary.pressure_face.row_idx[0], 3);
    EXPECT_EQ(boundary.pressure_face.col_idx[0][0], 3);
    EXPECT_NEAR(boundary.pressure_face.values[0][0], 1.0, 1e-10);

    EXPECT_TRUE(boundary.pressure_cell.row_idx.empty());
}

// ============================================================================
// Unit tests for tpfa_detail::add_neumann_boundary_entries
// ============================================================================

TEST(NeumannHelpers, BoundaryEntries)
{
    // face=5, cell=1, sign=-1, trm=4.0, vec=[0.5,0,0]
    FaceSideData side;
    side.face_ind = 5;
    side.cell_ind = 1;
    side.sign = -1;
    side.trm = 4.0;
    side.face_cell_vec = {0.5, 0.0, 0.0};

    BoundaryStencilData boundary;
    add_neumann_boundary_entries(boundary, side);

    // bound_flux: value = sign = -1
    ASSERT_EQ(boundary.bound_flux.row_idx.size(), 1u);
    EXPECT_EQ(boundary.bound_flux.row_idx[0], 5);
    EXPECT_EQ(boundary.bound_flux.col_idx[0][0], 5);
    EXPECT_NEAR(boundary.bound_flux.values[0][0], -1.0, 1e-10);

    // pressure_cell: value = 1, col = cell_ind
    ASSERT_EQ(boundary.pressure_cell.row_idx.size(), 1u);
    EXPECT_EQ(boundary.pressure_cell.col_idx[0][0], 1);
    EXPECT_NEAR(boundary.pressure_cell.values[0][0], 1.0, 1e-10);

    // pressure_face: value = -1/trm = -0.25
    ASSERT_EQ(boundary.pressure_face.row_idx.size(), 1u);
    EXPECT_EQ(boundary.pressure_face.col_idx[0][0], 5);
    EXPECT_NEAR(boundary.pressure_face.values[0][0], -0.25, 1e-10);

    // vector_source: col = {cell*3+0, cell*3+1, cell*3+2} = {3,4,5}, val = {0.5, 0, 0}
    ASSERT_EQ(boundary.vector_source.row_idx.size(), 1u);
    ASSERT_EQ(boundary.vector_source.col_idx[0].size(), 3u);
    EXPECT_EQ(boundary.vector_source.col_idx[0][0], 3);
    EXPECT_EQ(boundary.vector_source.col_idx[0][1], 4);
    EXPECT_EQ(boundary.vector_source.col_idx[0][2], 5);
    EXPECT_NEAR(boundary.vector_source.values[0][0], 0.5, 1e-10);
    EXPECT_NEAR(boundary.vector_source.values[0][1], 0.0, 1e-10);
    EXPECT_NEAR(boundary.vector_source.values[0][2], 0.0, 1e-10);

    // No flux entry for Neumann.
    EXPECT_TRUE(boundary.pressure_cell.row_idx.size() == 1u);
}

// ============================================================================
// Unit tests for tpfa_detail::accumulate_internal_face
// ============================================================================

TEST(AccumulateInternalFace, TransmissibilityAndVectorSource)
{
    // side_a: face=0, cell=0, sign=1, trm=4.0, vec=[-0.5,0,0]
    // side_b: face=0, cell=1, sign=-1, trm=6.0, vec=[0.5,0,0]
    // harmonic_mean = 4*6/(4+6) = 2.4
    FaceSideData side_a;
    side_a.face_ind = 0;
    side_a.cell_ind = 0;
    side_a.sign = 1;
    side_a.trm = 4.0;
    side_a.face_cell_vec = {-0.5, 0.0, 0.0};

    FaceSideData side_b;
    side_b.face_ind = 0;
    side_b.cell_ind = 1;
    side_b.sign = -1;
    side_b.trm = 6.0;
    side_b.face_cell_vec = {0.5, 0.0, 0.0};

    auto flux = init_tpfa_flux_stencil(1);
    auto boundary = init_tpfa_boundary_stencil(0);
    TpfaAccumulator acc{flux, boundary};

    accumulate_internal_face(acc, side_a, side_b);

    const double hm = 2.4;
    ASSERT_EQ(flux.row_idx.size(), 1u);
    EXPECT_EQ(flux.row_idx[0], 0);
    ASSERT_EQ(flux.col_idx[0].size(), 2u);
    EXPECT_EQ(flux.col_idx[0][0], 0);
    EXPECT_EQ(flux.col_idx[0][1], 1);
    EXPECT_NEAR(flux.flux_values[0][0], hm * 1, 1e-10);
    EXPECT_NEAR(flux.flux_values[0][1], hm * -1, 1e-10);

    ASSERT_EQ(flux.vs_values[0].size(), 6u);
    EXPECT_NEAR(flux.vs_values[0][0], hm * 1 * (-0.5), 1e-10);
    EXPECT_NEAR(flux.vs_values[0][1], 0.0, 1e-10);
    EXPECT_NEAR(flux.vs_values[0][2], 0.0, 1e-10);
    EXPECT_NEAR(flux.vs_values[0][3], hm * (-1) * 0.5, 1e-10);
    EXPECT_NEAR(flux.vs_values[0][4], 0.0, 1e-10);
    EXPECT_NEAR(flux.vs_values[0][5], 0.0, 1e-10);

    // No boundary entries for internal faces.
    EXPECT_TRUE(boundary.bound_flux.row_idx.empty());
}

// ============================================================================
// Unit tests for tpfa_detail::accumulate_boundary_face
// ============================================================================

TEST(AccumulateBoundaryFace, Dirichlet)
{
    FaceSideData side;
    side.face_ind = 3;
    side.cell_ind = 0;
    side.sign = 1;
    side.trm = 4.0;
    side.face_cell_vec = {-0.5, 0.0, 0.0};

    auto flux = init_tpfa_flux_stencil(1);
    auto boundary = init_tpfa_boundary_stencil(1);
    TpfaAccumulator acc{flux, boundary};

    accumulate_boundary_face(acc, side, BoundaryCondition::Dirichlet);

    ASSERT_EQ(flux.row_idx.size(), 1u);
    EXPECT_EQ(flux.row_idx[0], 3);
    EXPECT_NEAR(flux.flux_values[0][0], 4.0, 1e-10);

    ASSERT_EQ(boundary.bound_flux.row_idx.size(), 1u);
    EXPECT_NEAR(boundary.bound_flux.values[0][0], -4.0, 1e-10);

    ASSERT_EQ(boundary.pressure_face.row_idx.size(), 1u);
    EXPECT_NEAR(boundary.pressure_face.values[0][0], 1.0, 1e-10);

    EXPECT_TRUE(boundary.pressure_cell.row_idx.empty());
}

TEST(AccumulateBoundaryFace, Neumann)
{
    FaceSideData side;
    side.face_ind = 5;
    side.cell_ind = 1;
    side.sign = -1;
    side.trm = 4.0;
    side.face_cell_vec = {0.5, 0.0, 0.0};

    auto flux = init_tpfa_flux_stencil(1);
    auto boundary = init_tpfa_boundary_stencil(1);
    TpfaAccumulator acc{flux, boundary};

    accumulate_boundary_face(acc, side, BoundaryCondition::Neumann);

    // Neumann: no flux entry.
    EXPECT_TRUE(flux.row_idx.empty());
    EXPECT_TRUE(flux.col_idx.empty());

    ASSERT_EQ(boundary.bound_flux.row_idx.size(), 1u);
    EXPECT_NEAR(boundary.bound_flux.values[0][0], -1.0, 1e-10);

    ASSERT_EQ(boundary.pressure_cell.row_idx.size(), 1u);
    EXPECT_EQ(boundary.pressure_cell.col_idx[0][0], 1);
    EXPECT_NEAR(boundary.pressure_cell.values[0][0], 1.0, 1e-10);

    ASSERT_EQ(boundary.pressure_face.row_idx.size(), 1u);
    EXPECT_NEAR(boundary.pressure_face.values[0][0], -0.25, 1e-10);
}

TEST(AccumulateBoundaryFace, RobinThrows)
{
    FaceSideData side;
    side.face_ind = 0;
    side.cell_ind = 0;
    side.sign = 1;
    side.trm = 1.0;
    side.face_cell_vec = {0.5, 0.0, 0.0};

    auto flux = init_tpfa_flux_stencil(1);
    auto boundary = init_tpfa_boundary_stencil(1);
    TpfaAccumulator acc{flux, boundary};

    EXPECT_THROW(accumulate_boundary_face(acc, side, BoundaryCondition::Robin),
                 std::logic_error);
}

// ============================================================================
// Integration test fixture — 2×1 Cartesian grid, 2.0×1.0 domain.
// Face layout (from Python inspection):
//   0: left x-boundary, cell 0, sign=-1  (Dirichlet)
//   1: internal x-face, cells {0,1}
//   2: right x-boundary, cell 1, sign=+1 (Dirichlet)
//   3: bottom y-face, cell 0, sign=-1    (Neumann)
//   4: bottom y-face, cell 1, sign=-1    (Neumann)
//   5: top y-face, cell 0, sign=+1       (Neumann)
//   6: top y-face, cell 1, sign=+1       (Neumann)
// Tensor: isotropic k=2, cell-face distance = 0.5 in both x and y.
// nKproj(face_normal=[1,0], vec=[±0.5,0,0], sign=∓1) = 2*0.5/0.25 = 4.0
// nKproj(face_normal=[0,1], vec=[0,±0.5,0], sign=∓1) = 2*0.5/0.25 = 4.0
// ============================================================================

class TpfaCart2x1 : public ::testing::Test
{
   protected:
    std::unique_ptr<Grid> grid;
    ScalarDiscretization discr;

    void SetUp() override
    {
        grid = Grid::create_cartesian_grid(2, {2, 1}, {2.0, 1.0});
        grid->compute_geometry();

        SecondOrderTensor k(2, 2, {2.0, 2.0});
        std::unordered_map<int, BoundaryCondition> bc;
        bc[0] = BoundaryCondition::Dirichlet;
        bc[2] = BoundaryCondition::Dirichlet;
        bc[3] = BoundaryCondition::Neumann;
        bc[4] = BoundaryCondition::Neumann;
        bc[5] = BoundaryCondition::Neumann;
        bc[6] = BoundaryCondition::Neumann;

        discr = tpfa(*grid, k, bc);
    }
};

TEST_F(TpfaCart2x1, FluxOnBoundaryFaces)
{
    // Face 0 (Dirichlet, sign=-1): flux = trm*sign = 4.0*(-1) = -4.0
    EXPECT_NEAR(discr.flux->value(0, 0), -4.0, 1e-10);
    // Face 2 (Dirichlet, sign=+1): flux = trm*sign = 4.0*1 = 4.0
    EXPECT_NEAR(discr.flux->value(2, 1), 4.0, 1e-10);
    // Neumann faces: no cell contribution → 0
    EXPECT_NEAR(discr.flux->value(3, 0), 0.0, 1e-10);
    EXPECT_NEAR(discr.flux->value(5, 0), 0.0, 1e-10);
}

TEST_F(TpfaCart2x1, FluxOnInternalFace)
{
    // Face 1 (internal): harmonic_mean = 4*4/(4+4) = 2.0
    // side_a (cell 0, sign=+1), side_b (cell 1, sign=-1)
    EXPECT_NEAR(discr.flux->value(1, 0), 2.0, 1e-10);
    EXPECT_NEAR(discr.flux->value(1, 1), -2.0, 1e-10);
}

TEST_F(TpfaCart2x1, BoundFluxDirichlet)
{
    // bound_flux = -(trm * sign)
    EXPECT_NEAR(discr.bound_flux->value(0, 0), 4.0, 1e-10);   // -(4*-1) = 4
    EXPECT_NEAR(discr.bound_flux->value(2, 2), -4.0, 1e-10);  // -(4*1) = -4
}

TEST_F(TpfaCart2x1, BoundFluxNeumann)
{
    // bound_flux = sign (for Neumann)
    EXPECT_NEAR(discr.bound_flux->value(3, 3), -1.0, 1e-10);  // sign=-1
    EXPECT_NEAR(discr.bound_flux->value(5, 5), 1.0, 1e-10);   // sign=+1
}

TEST_F(TpfaCart2x1, VectorSourceInternalFace)
{
    // Face 1 internal: side_a cell=0, vec=[0.5,0,0]; side_b cell=1, vec=[-0.5,0,0]
    // harmonic_mean=2.0, sign_a=1, sign_b=-1
    // vs_values = [2*1*0.5, 0, 0, 2*(-1)*(-0.5), 0, 0] = [1.0, 0, 0, 1.0, 0, 0]
    // CSR col for cell 0, dim 0 = 0*3+0 = 0; for cell 1, dim 0 = 1*3+0 = 3
    EXPECT_NEAR(discr.vector_source->value(1, 0), 1.0, 1e-10);
    EXPECT_NEAR(discr.vector_source->value(1, 1), 0.0, 1e-10);
    EXPECT_NEAR(discr.vector_source->value(1, 2), 0.0, 1e-10);
    EXPECT_NEAR(discr.vector_source->value(1, 3), 1.0, 1e-10);
    EXPECT_NEAR(discr.vector_source->value(1, 4), 0.0, 1e-10);
    EXPECT_NEAR(discr.vector_source->value(1, 5), 0.0, 1e-10);
}

TEST_F(TpfaCart2x1, VectorSourceBoundaryDirichlet)
{
    // Face 0 Dirichlet: trm=4.0, sign=-1, vec=[-0.5,0,0]
    // vs = trm*sign*vec = 4*(-1)*[-0.5,0,0] = [2,0,0]
    // CSR col = cell_0 * 3 + k = 0,1,2
    EXPECT_NEAR(discr.vector_source->value(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(discr.vector_source->value(0, 1), 0.0, 1e-10);
    EXPECT_NEAR(discr.vector_source->value(0, 2), 0.0, 1e-10);
}

TEST_F(TpfaCart2x1, BoundPressureCellNeumann)
{
    // Neumann face 3: pressure_cell col = cell 0, value = 1
    EXPECT_NEAR(discr.bound_pressure_cell->value(3, 0), 1.0, 1e-10);
    // Neumann face 4: pressure_cell col = cell 1, value = 1
    EXPECT_NEAR(discr.bound_pressure_cell->value(4, 1), 1.0, 1e-10);
}

TEST_F(TpfaCart2x1, BoundPressureFace)
{
    // Dirichlet: pressure_face = 1 on diagonal
    EXPECT_NEAR(discr.bound_pressure_face->value(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(discr.bound_pressure_face->value(2, 2), 1.0, 1e-10);
    // Neumann: pressure_face = -1/trm = -1/4 = -0.25
    EXPECT_NEAR(discr.bound_pressure_face->value(3, 3), -0.25, 1e-10);
    EXPECT_NEAR(discr.bound_pressure_face->value(5, 5), -0.25, 1e-10);
}

TEST_F(TpfaCart2x1, BoundVectorSourceNeumann)
{
    // Face 3 Neumann: cell=0, sign=-1, vec=[0,-0.5,0] (face_center=[0.5,0]-cell_center=[0.5,0.5])
    // vector_source col = {0*3+0, 0*3+1, 0*3+2} = {0,1,2}, val = {0, -0.5, 0}
    EXPECT_NEAR(discr.bound_pressure_vector_source->value(3, 0), 0.0, 1e-10);
    EXPECT_NEAR(discr.bound_pressure_vector_source->value(3, 1), -0.5, 1e-10);
    EXPECT_NEAR(discr.bound_pressure_vector_source->value(3, 2), 0.0, 1e-10);
}

TEST_F(TpfaCart2x1, MatrixDimensions)
{
    EXPECT_EQ(discr.flux->num_rows(), 7);
    EXPECT_EQ(discr.flux->num_cols(), 2);
    EXPECT_EQ(discr.vector_source->num_rows(), 7);
    EXPECT_EQ(discr.vector_source->num_cols(), 6);  // 2 cells × 3
    EXPECT_EQ(discr.bound_flux->num_rows(), 7);
    EXPECT_EQ(discr.bound_flux->num_cols(), 7);
    EXPECT_EQ(discr.bound_pressure_cell->num_rows(), 7);
    EXPECT_EQ(discr.bound_pressure_cell->num_cols(), 2);
    EXPECT_EQ(discr.bound_pressure_face->num_rows(), 7);
    EXPECT_EQ(discr.bound_pressure_face->num_cols(), 7);
    EXPECT_EQ(discr.bound_pressure_vector_source->num_rows(), 7);
    EXPECT_EQ(discr.bound_pressure_vector_source->num_cols(), 6);  // 2 cells × 3
}

// ============================================================================
// Integration test: 3D grid, all-Dirichlet BCs.
// ============================================================================

TEST(TpfaStandalone, Cart3dAllDirichlet)
{
    auto g = Grid::create_cartesian_grid(3, {2, 2, 1}, {1.0, 1.0, 1.0});
    g->compute_geometry();

    SecondOrderTensor k(3, 4, {1.0, 1.0, 1.0, 1.0});
    std::unordered_map<int, BoundaryCondition> bc;
    for (int f{0}; f < g->num_faces(); ++f)
    {
        bc[f] = BoundaryCondition::Dirichlet;
    }
    // Mark internal faces only — need to identify them.
    // For a 2x2x1 grid, faces with exactly 2 adjacent cells are internal.
    for (int f{0}; f < g->num_faces(); ++f)
    {
        if (g->cells_of_face(f).size() == 2)
            bc.erase(f);
    }

    auto d = tpfa(*g, k, bc);

    EXPECT_EQ(d.flux->num_rows(), g->num_faces());
    EXPECT_EQ(d.flux->num_cols(), g->num_cells());
    EXPECT_EQ(d.vector_source->num_rows(), g->num_faces());
    EXPECT_EQ(d.vector_source->num_cols(), g->num_cells() * 3);
}

// ============================================================================
// Existing integration tests (unchanged)
// ============================================================================

// Test fixture for TPFA
class TPFA : public ::testing::Test
{
   protected:
    std::unique_ptr<Grid> grid;
    SecondOrderTensor tensor;
    ScalarDiscretization discr;
    std::unordered_map<int, BoundaryCondition> bc_map;

    // Constructor. Create a dummy tensor and empty discretization.
    TPFA() : grid(nullptr), tensor(2, 1, {1.0}), discr(), bc_map() {}

    void SetUp() override
    {
        // Create a simple grid. Extension of the grid is 1.0 in x and 2.0 in y.
        std::vector<int> num_cells = {2, 2};
        std::vector<double> lengths = {1.0, 2.0};
        grid = Grid::create_cartesian_grid(2, num_cells, lengths);
        grid->compute_geometry();

        tensor = SecondOrderTensor(2, 4, {1.0, 2.0, 3.0, 4.0});

        // Set Dirichlet conditions on the left and right boundaries, face indices {0,
        // 2, 3, 5}.
        bc_map[0] = BoundaryCondition::Dirichlet;
        bc_map[2] = BoundaryCondition::Dirichlet;
        bc_map[3] = BoundaryCondition::Dirichlet;
        bc_map[5] = BoundaryCondition::Dirichlet;
        // Set Neumann conditions on the top and bottom boundaries, face indices {6, 7, 10, 11}.
        bc_map[6] = BoundaryCondition::Neumann;
        bc_map[7] = BoundaryCondition::Neumann;
        bc_map[10] = BoundaryCondition::Neumann;
        bc_map[11] = BoundaryCondition::Neumann;

        // Compute the discretization
        discr = tpfa(*grid, tensor, bc_map);
    }
};

// Test that a Robin boundary condition throws std::logic_error.
TEST_F(TPFA, RobinBoundaryThrows)
{
    std::unordered_map<int, BoundaryCondition> robin_bc_map;
    for (int f{0}; f < grid->num_faces(); ++f)
    {
        robin_bc_map[f] = BoundaryCondition::Robin;
    }
    EXPECT_THROW(tpfa(*grid, tensor, robin_bc_map), std::logic_error);
}

// Test that a 1-cell 2D grid with all-Dirichlet BCs produces nonzero transmissibilities
// for each boundary face and matching bound_flux entries.
TEST(TpfaStandalone, SingleCell2dAllDirichlet)
{
    // 2D grid: 1 cell, unit square. All 4 faces are boundary faces.
    auto g = Grid::create_cartesian_grid(2, {1, 1}, {1.0, 1.0});
    g->compute_geometry();

    // Isotropic tensor, k=3, one cell.
    SecondOrderTensor k(2, 1, {3.0});

    std::unordered_map<int, BoundaryCondition> bc;
    for (int f{0}; f < g->num_faces(); ++f)
    {
        bc[f] = BoundaryCondition::Dirichlet;
    }

    auto d = tpfa(*g, k, bc);

    // For each boundary face the flux entry for cell 0 must be nonzero, and the
    // corresponding bound_flux entry (same face index) must have the opposite sign.
    for (int f{0}; f < g->num_faces(); ++f)
    {
        const double t = d.flux->value(f, 0);
        EXPECT_NE(t, 0.0) << "face " << f << " has zero flux transmissibility";
        EXPECT_NEAR(d.bound_flux->value(f, f), -t, 1e-10)
            << "face " << f << " bound_flux sign mismatch";
    }
}

// Test that Neumann BCs on all faces of a 2-cell 1D grid produce zero flux-matrix
// transmissibilities and unit bound-flux transmissibilities.
TEST(TpfaStandalone, NeumannBoundaryFlux)
{
    auto g = Grid::create_cartesian_grid(1, {2}, {1.0});
    g->compute_geometry();

    SecondOrderTensor k(1, 2, {1.0, 1.0});

    std::unordered_map<int, BoundaryCondition> bc;
    bc[0] = BoundaryCondition::Neumann;
    bc[2] = BoundaryCondition::Neumann;

    auto d = tpfa(*g, k, bc);

    // Boundary Neumann faces contribute nothing to the cell-pressure flux matrix.
    EXPECT_NEAR(d.flux->value(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(d.flux->value(2, 1), 0.0, 1e-10);

    // bound_flux should have magnitude 1 for each Neumann face.
    EXPECT_NEAR(std::abs(d.bound_flux->value(0, 0)), 1.0, 1e-10);
    EXPECT_NEAR(std::abs(d.bound_flux->value(2, 2)), 1.0, 1e-10);
}

// Test the flux values on a Cartesian 2d grid.
TEST_F(TPFA, FluxValuesCart2d)
{
    // Distance between the cell centers in the two directions.
    const double dx = 0.5;
    const double dy = 1.0;

    // Expected flux values for the internal faces. The nominator represents the area
    // of the face. The denominator represents the harmonic mean of the permeability
    // in the two cells adjacent to the face, divided by the distance to the face.
    // The factor 0.5 is needed to get the face-cell distance.
    const double t_1 = dy / (0.5 * dx / 1.0 + 0.5 * dx / 2.0);
    const double t_4 = dy / (0.5 * dx / 3.0 + 0.5 * dx / 4.0);
    const double t_8 = dx / (0.5 * dy / 1.0 + 0.5 * dy / 3.0);
    const double t_9 = dx / (0.5 * dy / 2.0 + 0.5 * dy / 4.0);

    // Check the flux values on internal faces.
    EXPECT_NEAR(discr.flux->value(1, 0), t_1, 1e-10);
    EXPECT_NEAR(discr.flux->value(1, 1), -t_1, 1e-10);
    EXPECT_NEAR(discr.flux->value(4, 2), t_4, 1e-10);
    EXPECT_NEAR(discr.flux->value(4, 3), -t_4, 1e-10);
    EXPECT_NEAR(discr.flux->value(8, 0), t_8, 1e-10);
    EXPECT_NEAR(discr.flux->value(8, 2), -t_8, 1e-10);
    EXPECT_NEAR(discr.flux->value(9, 1), t_9, 1e-10);
    EXPECT_NEAR(discr.flux->value(9, 3), -t_9, 1e-10);

    // Expected flux for the boundary faces. On the lateral faces, a Dirichlet condition
    // is set. We need a minus sign on the faces on the left side, since the respective
    // face normal points into the cell.
    const double t_0 = -dy / (0.5 * dx / 1.0);
    const double t_2 = dy / (0.5 * dx / 2.0);
    const double t_3 = -dy / (0.5 * dx / 3.0);
    const double t_5 = dy / (0.5 * dx / 4.0);
    // On the top and bottom faces, a Neumann condition is set, which means that the
    // transmissibility should be 0.
    const double t_6 = 0.0;
    const double t_7 = 0.0;
    const double t_10 = 0.0;
    const double t_11 = 0.0;
    // Check the flux values on boundary faces.
    EXPECT_NEAR(discr.flux->value(0, 0), t_0, 1e-10);
    EXPECT_NEAR(discr.flux->value(2, 1), t_2, 1e-10);
    EXPECT_NEAR(discr.flux->value(3, 2), t_3, 1e-10);
    EXPECT_NEAR(discr.flux->value(5, 3), t_5, 1e-10);
    EXPECT_NEAR(discr.flux->value(6, 0), t_6, 1e-10);
    EXPECT_NEAR(discr.flux->value(7, 1), t_7, 1e-10);
    EXPECT_NEAR(discr.flux->value(10, 2), t_10, 1e-10);
    EXPECT_NEAR(discr.flux->value(11, 3), t_11, 1e-10);

    // Finally, check the discretization of boundary conditions. On the Dirichlet faces,
    // the boundary discretization should be the negative of the transmissibility of the
    // corresponding cell.
    EXPECT_NEAR(discr.bound_flux->value(0, 0), -t_0, 1e-10);
    EXPECT_NEAR(discr.bound_flux->value(2, 2), -t_2, 1e-10);
    EXPECT_NEAR(discr.bound_flux->value(3, 3), -t_3, 1e-10);
    EXPECT_NEAR(discr.bound_flux->value(5, 5), -t_5, 1e-10);
    // On the Neumann faces, the boundary discretization should be 1 if the face normal
    // points out of the cell, and -1 if it points into the cell.
    EXPECT_NEAR(discr.bound_flux->value(6, 6), -1.0, 1e-10);
    EXPECT_NEAR(discr.bound_flux->value(7, 7), -1.0, 1e-10);
    EXPECT_NEAR(discr.bound_flux->value(10, 10), 1.0, 1e-10);
    EXPECT_NEAR(discr.bound_flux->value(11, 11), 1.0, 1e-10);

    // Check the size of the boundary face pressure reconstruction matrices.
    EXPECT_EQ(discr.bound_pressure_cell->num_rows(), 12);
    EXPECT_EQ(discr.bound_pressure_cell->num_cols(), 4);  // 4 cells.
    EXPECT_EQ(discr.bound_pressure_face->num_rows(), 12);
    EXPECT_EQ(discr.bound_pressure_face->num_cols(), 12);  // 12 faces.

    // Check the size of the vector source and bound vector source.
    EXPECT_EQ(discr.vector_source->num_rows(), 12);
    EXPECT_EQ(discr.vector_source->num_cols(), 4 * 3);  // 3 components per cell.
    EXPECT_EQ(discr.bound_pressure_vector_source->num_rows(), 12);
    EXPECT_EQ(discr.bound_pressure_vector_source->num_cols(), 4 * 3);  // 3 components per cell.
}