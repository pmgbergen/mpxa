#include <gtest/gtest.h>

#include "../../src/grid/grid.cpp"

// Test fixture for Grid
class GridTest : public ::testing::Test
{
   protected:
    const int num_cells_2d[2] = {2, 3};
    const double lengths_2d[2] = {2.0, 3.0};
    const double unit_lengths_2d[2] = {1.0, 1.0};
    Grid* grid_2d;
    Grid* unit_square;

    void SetUp() override
    {
        grid_2d = create_cartesian_grid(2, num_cells_2d, lengths_2d);
        unit_square = create_cartesian_grid(2, num_cells_2d, unit_lengths_2d);
    }

    void TearDown() override
    {
        delete grid_2d;
        delete unit_square;
    }
};

// Test that the grid nodes are created correctly
TEST_F(GridTest, Nodes)
{
    EXPECT_EQ(grid_2d->num_nodes(), 12);

    const double** nodes = grid_2d->nodes();
    EXPECT_EQ(nodes[0][0], 0.0);
    EXPECT_EQ(nodes[0][1], 0.0);
    EXPECT_EQ(nodes[1][0], 1.0);
    EXPECT_EQ(nodes[1][1], 0.0);
    EXPECT_EQ(nodes[2][0], 2.0);
    EXPECT_EQ(nodes[2][1], 0.0);
    EXPECT_EQ(nodes[3][0], 0.0);
    EXPECT_EQ(nodes[3][1], 1.0);
    EXPECT_EQ(nodes[4][0], 1.0);
    EXPECT_EQ(nodes[4][1], 1.0);
    EXPECT_EQ(nodes[5][0], 2.0);
    EXPECT_EQ(nodes[5][1], 1.0);
    EXPECT_EQ(nodes[6][0], 0.0);
    EXPECT_EQ(nodes[6][1], 2.0);
    EXPECT_EQ(nodes[7][0], 1.0);
    EXPECT_EQ(nodes[7][1], 2.0);
    EXPECT_EQ(nodes[8][0], 2.0);
    EXPECT_EQ(nodes[8][1], 2.0);
    EXPECT_EQ(nodes[9][0], 0.0);
    EXPECT_EQ(nodes[9][1], 3.0);
    EXPECT_EQ(nodes[10][0], 1.0);
    EXPECT_EQ(nodes[10][1], 3.0);
    EXPECT_EQ(nodes[11][0], 2.0);
    EXPECT_EQ(nodes[11][1], 3.0);
}
