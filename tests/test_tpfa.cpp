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

    TPFA() : grid(nullptr), tensor(2, 4, new double[4]{1.0, 2.0, 3.0, 4.0}), discr() {}

    void SetUp() override
    {
        // Create a simple grid. Extension of the grid is 1.0 in x and 2.0 in y.
        int* num_cells = new int[2]{2, 2};
        double* lengths = new double[2]{1.0, 2.0};
        grid = Grid::create_cartesian_grid(2, num_cells, lengths);
        grid->compute_geometry();
        delete[] num_cells;
        delete[] lengths;

        // Compute the discretization
        discr = tpfa(*grid, tensor);
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

    // Check the flux values.
    EXPECT_EQ(discr.flux->value(1, 0), t_1);
    EXPECT_EQ(discr.flux->value(1, 1), -t_1);
    EXPECT_EQ(discr.flux->value(4, 2), t_4);
    EXPECT_EQ(discr.flux->value(4, 3), -t_4);
    EXPECT_EQ(discr.flux->value(8, 0), t_8);
    EXPECT_EQ(discr.flux->value(8, 2), -t_8);
    EXPECT_EQ(discr.flux->value(9, 1), t_9);
    EXPECT_EQ(discr.flux->value(9, 3), -t_9);
}