#include <gtest/gtest.h>

#include <numeric>

#include "../../include/discr.h"
#include "../../include/grid.h"
#include "../../include/tensor.h"

// Test fixture for MPFA
class MPFA : public ::testing::Test
{
   protected:
    std::unique_ptr<Grid> grid_2d;
    ScalarDiscretization discr_2d;
    std::unordered_map<int, BoundaryCondition> bc_map_2d;

    // Constructor. Create a dummy tensor and empty discretization.
    MPFA() : grid_2d(nullptr), discr_2d(), bc_map_2d() {}

    void SetUp() override
    {
        // Create a simple grid. Let it be 3x3
        std::vector<int> num_cells = {3, 3};
        std::vector<double> lengths = {3.0, 6.0};
        grid_2d = Grid::create_cartesian_grid(2, num_cells, lengths);
        grid_2d->compute_geometry();

        // Set the tensor to be isotropic.

        // Set Dirichlet conditions on the left and right boundaries, face indices {0,
        // 2, 3, 5}.
        bc_map_2d[0] = BoundaryCondition::Dirichlet;
        bc_map_2d[3] = BoundaryCondition::Dirichlet;
        bc_map_2d[4] = BoundaryCondition::Dirichlet;
        bc_map_2d[7] = BoundaryCondition::Dirichlet;
        bc_map_2d[8] = BoundaryCondition::Dirichlet;
        bc_map_2d[11] = BoundaryCondition::Dirichlet;

        // Set Neumann conditions on the top and bottom boundaries, face indices {6, 7, 10, 11}.
        bc_map_2d[12] = BoundaryCondition::Neumann;
        bc_map_2d[13] = BoundaryCondition::Neumann;
        bc_map_2d[14] = BoundaryCondition::Neumann;
        bc_map_2d[21] = BoundaryCondition::Neumann;
        bc_map_2d[22] = BoundaryCondition::Neumann;
        bc_map_2d[23] = BoundaryCondition::Neumann;
    }
};

void test_flux_values(const ScalarDiscretization& discr, int face_ind,
                      const std::vector<int>& cell_indices, const std::vector<double>& values)
{
    for (size_t i = 0; i < cell_indices.size(); ++i)
    {
        // Check the flux values for the internal faces.
        EXPECT_NEAR(discr.flux->value(face_ind, cell_indices[i]), values[i], 1e-6);
    }
}

void test_bound_flux_values(const ScalarDiscretization& discr, int face_ind,
                            const std::vector<int>& face_indices, const std::vector<double>& values)
{
    for (size_t i = 0; i < face_indices.size(); ++i)
    {
        // Check the flux values for the internal faces.
        EXPECT_NEAR(discr.bound_flux->value(face_ind, face_indices[i]), values[i], 1e-6);
    }
}

TEST_F(MPFA, FluxValuesInternalFace2dIsotropic)
{
    // Set tensor, construct discretization.
    std::vector<double> k_xx(grid_2d->num_cells());
    std::iota(k_xx.begin(), k_xx.end(), 1.0);  // Fill with 1.0, 2.0, ..., num_cells

    SecondOrderTensor K = SecondOrderTensor(2, 9, k_xx);
    // Compute the discretization
    discr_2d = mpfa(*grid_2d, K, bc_map_2d);

    // Distance between the cell centers in the two directions.
    const double dx = 1.0;
    const double dy = 2.0;

    // Test the flux values on the internal faces 5 and 16.
    const double t_5 = dy / (0.5 * dx / 4.0 + 0.5 * dx / 5.0);
    const std::vector<int> cell_indices_face_5 = {0, 1, 3, 4, 6, 7};
    const std::vector<double> values_face_5 = {0.0, 0.0, t_5, -t_5, 0.0, 0.0};
    test_flux_values(discr_2d, 5, cell_indices_face_5, values_face_5);

    const double t_16 = dx / (0.5 * dy / 2.0 + 0.5 * dy / 5.0);
    const std::vector<int> cell_indices_face_16 = {0, 1, 2, 3, 4, 5};
    const std::vector<double> values_face_16 = {0.0, t_16, 0.0, 0.0, -t_16, 0.0};
    test_flux_values(discr_2d, 16, cell_indices_face_16, values_face_16);

    // Check the boundary flux values:

    // Face 0 is a Dirichlet boundary face in a corner of a grid.
    const double t_0 = dy / (0.5 * dx / 1.0);
    const std::vector<int> cell_indices_face_0 = {0, 4};
    const std::vector<double> values_cell_0 = {-t_0, 0.0};
    test_flux_values(discr_2d, 0, cell_indices_face_0, values_cell_0);
    const std::vector<int> face_indices_face_0 = {0, 4, 12, 15};
    const std::vector<double> values_face_0 = {t_0, 0.0, 0.0, 0.0};
    test_bound_flux_values(discr_2d, 0, face_indices_face_0, values_face_0);

    // // Face 4 is a Dirichlet boundary face in the middle of a grid. The permeability in
    // // the nearby cell is 4, so the flux is 4 times the value in cell 0.
    const double t_4 = dy / (0.5 * dx / 4.0);
    const std::vector<int> cell_indices_face_4 = {0, 3, 6};
    const std::vector<double> values_cell_4 = {0.0, -t_4, 0.0};
    test_flux_values(discr_2d, 4, cell_indices_face_4, values_cell_4);
    const std::vector<int> face_indices_face_4 = {0, 4, 8, 15, 18};
    const std::vector<double> values_face_4 = {-0.0, t_4, 0.0, 0.0, 0.0};
    test_bound_flux_values(discr_2d, 4, face_indices_face_4, values_face_4);

    // Face 12 is a Neumann boundary face in a corner of a grid.
    const std::vector<int> cell_indices_face_12 = {0, 1};
    const std::vector<double> values_cell_12 = {0.0, 0.0};
    test_flux_values(discr_2d, 12, cell_indices_face_12, values_cell_12);
    const std::vector<int> face_indices_face_12 = {0, 1, 12, 13};
    const std::vector<double> values_face_12 = {0.0, 0.0, -1.0, 0.0};
    test_bound_flux_values(discr_2d, 12, face_indices_face_12, values_face_12);

    // Face 13 is a Neumann boundary face in the middle of a grid.
    const std::vector<int> cell_indices_face_13 = {0, 1, 2};
    const std::vector<double> values_cell_13 = {0.0, 0.0, 0.0};
    test_flux_values(discr_2d, 13, cell_indices_face_13, values_cell_13);
    const std::vector<int> face_indices_face_13 = {1, 2, 12, 13, 14};
    const std::vector<double> values_face_13 = {0.0, 0.0, 0.0, -1.0, 0.0};
    test_bound_flux_values(discr_2d, 13, face_indices_face_13, values_face_13);
}

TEST_F(MPFA, FluxValuesInternalFace2dAnisotropic)
{
    std::vector<double> k_xx(grid_2d->num_cells());
    std::iota(k_xx.begin(), k_xx.end(), 1.0);             // Fill with 1.0, 2.0, ..., num_cells
    std::vector<double> k_yy(grid_2d->num_cells(), 0.0);  // Anisotropic tensor
    for (size_t i = 0; i < k_yy.size(); ++i)
    {
        k_yy[i] = 2.0 * (i + 1);
    }
    SecondOrderTensor K = SecondOrderTensor(2, 9, k_xx);
    K = K.with_kyy(k_yy);  // Set the y-component of the tensor

    // Compute the discretization
    discr_2d = mpfa(*grid_2d, K, bc_map_2d);
    // Distance between the cell centers in the two directions.
    const double dx = 1.0;
    const double dy = 2.0;

    const double t_5 = dy / (0.5 * dx / 4.0 + 0.5 * dx / 5.0);
    const std::vector<int> cell_indices_face_5 = {0, 1, 3, 4, 6, 7};
    const std::vector<double> values_face_5 = {0.0, 0.0, t_5, -t_5, 0.0, 0.0};
    test_flux_values(discr_2d, 5, cell_indices_face_5, values_face_5);

    const double t_16 = dx / (0.5 * dy / 4.0 + 0.5 * dy / 10.0);
    const std::vector<int> cell_indices_face_16 = {0, 1, 2, 3, 4, 5};
    const std::vector<double> values_face_16 = {0.0, t_16, 0.0, 0.0, -t_16, 0.0};
    test_flux_values(discr_2d, 16, cell_indices_face_16, values_face_16);
}

TEST_F(MPFA, FluxValuesInternalFace2dFullTensor)
{
    const double k_xx_value = 2.0;
    const double k_yy_value = 3.0;
    const double k_xy_value = 0.5;

    std::vector<double> k_xx(grid_2d->num_cells());
    std::fill(k_xx.begin(), k_xx.end(), k_xx_value);
    std::vector<double> k_yy(grid_2d->num_cells(), 0.0);
    std::fill(k_yy.begin(), k_yy.end(), k_yy_value);
    std::vector<double> k_xy(grid_2d->num_cells(), 0.0);
    std::fill(k_xy.begin(), k_xy.end(), k_xy_value);

    SecondOrderTensor K = SecondOrderTensor(2, 9, k_xx);
    K = K.with_kyy(k_yy);  // Set the y-component of the tensor
    K = K.with_kxy(k_xy);  // Set the xy-component of the tensor

    // Compute the discretization
    discr_2d = mpfa(*grid_2d, K, bc_map_2d);
    // Distance between the cell centers in the two directions.
    const double dx = 1.0;
    const double dy = 2.0;

    // Analytical expressions for the flux values on the internal faces.
    // Picked from Aavatsmark 2002, An introduction to multipoint flux approximation methods
    // on quadrilateral grids.
    // For the umtenth time, thanks your consistent the thoroughness, Ivar!
    const double a = k_xx_value * dy * dy / (dx * dy);
    const double b = k_yy_value * dx * dx / (dx * dy);
    const double c = k_xy_value * dx * dy / (dx * dy);

    const double t_5_0 = c / 4 * (1 + c / b);
    const double t_5_7 = -t_5_0;
    const double t_5_3 = a - c * c / (2 * b);
    const double t_5_4 = -t_5_3;
    const double t_5_6 = -c / 4 * (1 - c / b);
    const double t_5_1 = -t_5_6;

    const std::vector<int> cell_indices_face_5 = {0, 1, 3, 4, 6, 7};
    const std::vector<double> values_face_5 = {t_5_0, t_5_1, t_5_3, t_5_4, t_5_6, t_5_7};
    test_flux_values(discr_2d, 5, cell_indices_face_5, values_face_5);

    const double t_16_0 = c / 4 * (1 + c / a);
    const double t_16_5 = -t_16_0;
    const double t_16_1 = b - c * c / (2 * a);
    const double t_16_4 = -t_16_1;
    const double t_16_2 = -c / 4 * (1 - c / a);
    const double t_16_3 = -t_16_2;
    const std::vector<int> cell_indices_face_16 = {0, 1, 2, 3, 4, 5};
    const std::vector<double> values_face_16 = {t_16_0, t_16_1, t_16_2, t_16_3, t_16_4, t_16_5};
    test_flux_values(discr_2d, 16, cell_indices_face_16, values_face_16);
}

// MPFA should fall back to TPFA for 1D grids. Verify boundary face flux matrices match.
TEST(MpfaStandalone, FallsBackToTpfaIn1d)
{
    auto g = Grid::create_cartesian_grid(1, {2}, {1.0});
    g->compute_geometry();

    SecondOrderTensor k(1, 2, {1.0, 2.0});

    std::unordered_map<int, BoundaryCondition> bc;
    bc[0] = BoundaryCondition::Dirichlet;
    bc[2] = BoundaryCondition::Dirichlet;

    auto d_mpfa = mpfa(*g, k, bc);
    auto d_tpfa = tpfa(*g, k, bc);

    // Boundary face (Dirichlet) flux values should match. Face 0 → cell 0, face 2 → cell 1.
    EXPECT_NEAR(d_mpfa.flux->value(0, 0), d_tpfa.flux->value(0, 0), 1e-10);
    EXPECT_NEAR(d_mpfa.flux->value(2, 1), d_tpfa.flux->value(2, 1), 1e-10);
}

// Verify that the vector source matrix has the expected dimensions for a 2D grid.
TEST_F(MPFA, VectorSourceMatrixDimensions2d)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);

    EXPECT_EQ(discr_2d.vector_source->num_rows(), grid_2d->num_faces());
    EXPECT_EQ(discr_2d.vector_source->num_cols(), grid_2d->num_cells() * 3);
}

// Robin BC on any face must throw std::logic_error.
TEST_F(MPFA, RobinBoundaryThrows)
{
    std::unordered_map<int, BoundaryCondition> robin_bc_map;
    for (int f{0}; f < grid_2d->num_faces(); ++f)
    {
        robin_bc_map[f] = BoundaryCondition::Robin;
    }
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    EXPECT_THROW(mpfa(*grid_2d, k, robin_bc_map), std::exception);
}

// Verify that a 2D grid with a Dirichlet boundary face produces a nonzero bound_flux entry.
TEST_F(MPFA, BoundaryFaceFlux2dDirichlet)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    // Face 0 is a Dirichlet boundary; it should contribute a nonzero entry.
    EXPECT_NE(discr_2d.bound_flux->value(0, 0), 0.0);
}

// Verify that a 2D grid with a Neumann boundary face produces a nonzero bound_flux entry.
TEST_F(MPFA, BoundaryFaceFlux2dNeumann)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    // Face 12 is a Neumann boundary; it should contribute a nonzero entry.
    EXPECT_NE(discr_2d.bound_flux->value(12, 12), 0.0);
}

// Verify that with a Dirichlet boundary, bound_pressure_face has a 1/num_nodes entry.
TEST_F(MPFA, PressureReconstructionDirichletIsUnit)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    // For a 2D Cartesian grid, a Dirichlet boundary face has 2 nodes, so each of the two
    // interaction regions around the face nodes contributes 0.5, summing to 1.0.
    EXPECT_NEAR(discr_2d.bound_pressure_face->value(0, 0), 1.0, 1e-10);
}

// Verify that vector_source has SPATIAL_DIM * num_cells columns.
TEST_F(MPFA, VectorSourceColumnCount)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    constexpr int SPATIAL_DIM = 3;
    EXPECT_EQ(discr_2d.vector_source->num_cols(), SPATIAL_DIM * grid_2d->num_cells());
}

// ---------------------------------------------------------------------------
// Iteration 7 — new helpers: InteractionRegionGeometry, LocalBalanceMatrices,
// RowSortingInfo / compute_row_sorting, BoundaryOutputAccumulators,
// accumulate_boundary_data.
//
// These helpers live in an anonymous namespace in src/mpfa.cpp and are not
// accessible from the test translation unit. All tests below exercise them
// *indirectly* through the public mpfa() function. Each test targets a specific
// behavior that would fail (wrong values, wrong dimensions, or crash) if the
// corresponding helper were broken.
// ---------------------------------------------------------------------------

// --- bound_flux matrix dimensions ---
// compute_row_sorting + create_csr_matrix must produce num_rows = num_faces and
// num_cols = num_faces for bound_flux.
TEST_F(MPFA, BoundFluxMatrix_HasCorrectDimensions)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    EXPECT_EQ(discr_2d.bound_flux->num_rows(), grid_2d->num_faces());
    EXPECT_EQ(discr_2d.bound_flux->num_cols(), grid_2d->num_faces());
}

// --- bound_pressure_cell matrix dimensions ---
// accumulate_boundary_data must produce a (num_faces x num_cells) pressure-cell matrix.
TEST_F(MPFA, BoundPressureCellMatrix_HasCorrectDimensions)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    EXPECT_EQ(discr_2d.bound_pressure_cell->num_rows(), grid_2d->num_faces());
    EXPECT_EQ(discr_2d.bound_pressure_cell->num_cols(), grid_2d->num_cells());
}

// --- bound_pressure_face matrix dimensions ---
// accumulate_boundary_data must produce a (num_faces x num_faces) pressure-face matrix.
TEST_F(MPFA, BoundPressureFaceMatrix_HasCorrectDimensions)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    EXPECT_EQ(discr_2d.bound_pressure_face->num_rows(), grid_2d->num_faces());
    EXPECT_EQ(discr_2d.bound_pressure_face->num_cols(), grid_2d->num_faces());
}

// --- bound_pressure_vector_source matrix dimensions ---
// accumulate_boundary_data must produce a (num_faces x 3*num_cells) matrix.
// SPATIAL_DIM is always 3 inside mpfa.cpp.
TEST_F(MPFA, BoundPressureVectorSource_HasCorrectDimensions)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    constexpr int SPATIAL_DIM = 3;
    EXPECT_EQ(discr_2d.bound_pressure_vector_source->num_rows(), grid_2d->num_faces());
    EXPECT_EQ(discr_2d.bound_pressure_vector_source->num_cols(),
              SPATIAL_DIM * grid_2d->num_cells());
}

// --- Dirichlet path skips cell-pressure contribution ---
// In accumulate_boundary_data the Dirichlet branch hits `continue` before
// adding to pressure_reconstruction_cell_values.  Consequently the
// bound_pressure_cell matrix must have no entry for any Dirichlet face.
//
// Face 0 in bc_map_2d is Dirichlet; cell 0 is adjacent to it.
TEST_F(MPFA, BoundPressureCell_DirichletFaceHasZeroContribution)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    // bound_pressure_cell.value returns 0 for any cell because the Dirichlet
    // branch never writes into pressure_reconstruction_cell_values.
    for (int c = 0; c < grid_2d->num_cells(); ++c)
    {
        EXPECT_EQ(discr_2d.bound_pressure_cell->value(0, c), 0.0);
    }
}

// --- Neumann path adds cell-pressure contribution ---
// For a Neumann boundary face accumulate_boundary_data does NOT hit `continue`
// and therefore writes into pressure_reconstruction_cell_values for the
// adjacent cell.
//
// Face 12 in bc_map_2d is Neumann; it sits at the bottom-left corner of the
// 3×3 grid so cell 0 is adjacent.
TEST_F(MPFA, BoundPressureCell_NeumannFaceHasNonzeroContribution)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    // At least one cell must contribute to the pressure reconstruction of the
    // Neumann face.
    bool any_nonzero = false;
    for (int c = 0; c < grid_2d->num_cells(); ++c)
    {
        if (discr_2d.bound_pressure_cell->value(12, c) != 0.0)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

// --- Neumann face-pressure reconstruction is nonzero ---
// For a Neumann boundary face accumulate_boundary_data computes a face
// contribution via balance_faces_inv; the result must be nonzero.
//
// Face 13 is an interior Neumann face on the bottom boundary of the 3×3 grid.
TEST_F(MPFA, BoundPressureFace_NeumannFaceHasNonzeroEntries)
{
    SecondOrderTensor k(2, grid_2d->num_cells(), std::vector<double>(grid_2d->num_cells(), 1.0));
    discr_2d = mpfa(*grid_2d, k, bc_map_2d);
    bool any_nonzero = false;
    for (int f = 0; f < grid_2d->num_faces(); ++f)
    {
        if (discr_2d.bound_pressure_face->value(13, f) != 0.0)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

// --- make_local_balance_matrices with non-square interaction regions ---
// On a 2×3 grid each interaction region has a different number of local faces
// vs local cells.  make_local_balance_matrices(num_faces, num_cells) must
// produce correctly-dimensioned zero matrices; if it does not, the Eigen
// operations inside fill_cell_contributions / compute_local_flux will either
// crash (dimension mismatch) or produce wrong results.
//
// This standalone test also exercises compute_interaction_region_geometry and
// compute_row_sorting on a grid that is not used anywhere else.
TEST(MpfaStandalone, On2x3GridCompletesWithCorrectDimensions)
{
    auto g = Grid::create_cartesian_grid(2, {2, 3}, {2.0, 3.0});
    g->compute_geometry();

    SecondOrderTensor k(2, g->num_cells(), std::vector<double>(g->num_cells(), 1.0));

    // Programmatically mark all boundary faces as Dirichlet.
    std::unordered_map<int, BoundaryCondition> bc;
    for (const int f : g->boundary_faces())
    {
        bc[f] = BoundaryCondition::Dirichlet;
    }

    auto d = mpfa(*g, k, bc);

    // flux : num_faces × num_cells
    EXPECT_EQ(d.flux->num_rows(), g->num_faces());
    EXPECT_EQ(d.flux->num_cols(), g->num_cells());

    // bound_flux : num_faces × num_faces
    EXPECT_EQ(d.bound_flux->num_rows(), g->num_faces());
    EXPECT_EQ(d.bound_flux->num_cols(), g->num_faces());

    // bound_pressure_cell : num_faces × num_cells
    EXPECT_EQ(d.bound_pressure_cell->num_rows(), g->num_faces());
    EXPECT_EQ(d.bound_pressure_cell->num_cols(), g->num_cells());

    // bound_pressure_face : num_faces × num_faces
    EXPECT_EQ(d.bound_pressure_face->num_rows(), g->num_faces());
    EXPECT_EQ(d.bound_pressure_face->num_cols(), g->num_faces());
}

// --- RowSortingInfo: accumulated flux value for a face shared by multiple regions ---
// compute_row_sorting sorts contributions from all interaction regions that
// include a given face and accumulate them into a single CSR row.  For an
// interior face in the centre of a 3×3 isotropic unit-K grid the MPFA stencil
// equals the TPFA result (pure two-point), so the dominant entries are
// T = area / (dist_left + dist_right).  If compute_row_sorting mis-counted
// occurrences or sorted incorrectly, the accumulated row would contain
// duplicates or have the wrong value.
//
// Face 5 is the vertical internal face between cells 3 and 4 in the 3×3 grid.
// dx = 1 m, dy = 2 m, K_left = 4, K_right = 5 (from iota 1..9).
TEST_F(MPFA, RowSortingInfo_AccumulatedFluxIsCorrectForInternalFace)
{
    std::vector<double> k_xx(grid_2d->num_cells());
    std::iota(k_xx.begin(), k_xx.end(), 1.0);  // 1, 2, …, 9

    SecondOrderTensor K(2, grid_2d->num_cells(), k_xx);
    discr_2d = mpfa(*grid_2d, K, bc_map_2d);

    // Expected transmissibility for face 5 (vertical, dx=1, dy=2, K3=4, K4=5)
    const double dx = 1.0;
    const double dy = 2.0;
    const double t_5 = dy / (0.5 * dx / 4.0 + 0.5 * dx / 5.0);

    // Cell 3 and cell 4 are on opposite sides; signs must be opposite.
    EXPECT_NEAR(discr_2d.flux->value(5, 3), t_5, 1e-6);
    EXPECT_NEAR(discr_2d.flux->value(5, 4), -t_5, 1e-6);
    // The sum must be zero (flux is antisymmetric across an internal face for
    // these two dominant entries when the grid is isotropic diagonal).
    EXPECT_NEAR(discr_2d.flux->value(5, 3) + discr_2d.flux->value(5, 4), 0.0, 1e-10);
}

