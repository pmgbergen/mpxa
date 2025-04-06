#include <gtest/gtest.h>

#include "../src/fv/tpfa.cpp"
#include "../src/grid/grid.h"
#include "../src/utils/tensor.h"

// Test fixture for TPFA
class TPFA : public ::testing::Test
{
   protected:
    std::unique_ptr<Grid> grid;
    SecondOrderTensor tensor;
    ScalarDiscretization discr;
    std::map<int, BoundaryCondition> bc_map;

    // EK note to self: We can use the default constructor for the grid and discr. However,
    // the tensor contains unique_ptrs, and cannot be copied without a move constructor,
    // which is not implemneted. Hency it is easier to define the tensor in full in the
    // constructor of the test fixture.
    TPFA() : grid(nullptr), tensor(2, 4, new double[4]{1.0, 2.0, 3.0, 4.0}), discr(), bc_map() {}

    void SetUp() override
    {
        // Create a simple grid. Extension of the grid is 1.0 in x and 2.0 in y.
        std::vector<int> num_cells = {2, 2};
        std::vector<double> lengths = {1.0, 2.0};
        grid = Grid::create_cartesian_grid(2, num_cells, lengths);
        grid->compute_geometry();

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
    EXPECT_EQ(discr.flux->value(1, 0), t_1);
    EXPECT_EQ(discr.flux->value(1, 1), -t_1);
    EXPECT_EQ(discr.flux->value(4, 2), t_4);
    EXPECT_EQ(discr.flux->value(4, 3), -t_4);
    EXPECT_EQ(discr.flux->value(8, 0), t_8);
    EXPECT_EQ(discr.flux->value(8, 2), -t_8);
    EXPECT_EQ(discr.flux->value(9, 1), t_9);
    EXPECT_EQ(discr.flux->value(9, 3), -t_9);

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
    EXPECT_EQ(discr.flux->value(0, 0), t_0);
    EXPECT_EQ(discr.flux->value(2, 1), t_2);
    EXPECT_EQ(discr.flux->value(3, 2), t_3);
    EXPECT_EQ(discr.flux->value(5, 3), t_5);
    EXPECT_EQ(discr.flux->value(6, 0), t_6);
    EXPECT_EQ(discr.flux->value(7, 1), t_7);
    EXPECT_EQ(discr.flux->value(10, 2), t_10);
    EXPECT_EQ(discr.flux->value(11, 3), t_11);

    // Finally, check the discretization of boundary conditions. On the Dirichlet faces,
    // the boundary discretization should be the negative of the transmissibility of the
    // corresponding cell.
    EXPECT_EQ(discr.bound_flux->value(0, 0), -t_0);
    EXPECT_EQ(discr.bound_flux->value(2, 2), -t_2);
    EXPECT_EQ(discr.bound_flux->value(3, 3), -t_3);
    EXPECT_EQ(discr.bound_flux->value(5, 5), -t_5);
    // On the Neumann faces, the boundary discretization should be 1 if the face normal
    // points out of the cell, and -1 if it points into the cell.
    EXPECT_EQ(discr.bound_flux->value(6, 6), -1.0);
    EXPECT_EQ(discr.bound_flux->value(7, 7), -1.0);
    EXPECT_EQ(discr.bound_flux->value(10, 10), 1.0);
    EXPECT_EQ(discr.bound_flux->value(11, 11), 1.0);
}