#include <gtest/gtest.h>

#include <map>
#include <vector>

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

// Test that the grid nodes are created correctly. The nodes are created in a 2x3 grid
// with unit cell size.
TEST_F(GridTest, NodeCoordinates2dUnitCellSize)
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

// Test that grid nodes correct for a unit square domain (non-unit size grids). Only
// test a few nodes.
TEST_F(GridTest, NodeCoordinates2dUnitSquareDomain)
{
    const int nx = 2;
    const int ny = 3;

    const double** nodes = unit_square->nodes();
    EXPECT_EQ(nodes[0][0], 0.0);
    EXPECT_EQ(nodes[0][1], 0.0);
    EXPECT_EQ(nodes[1][0], 1.0 / nx);
    EXPECT_EQ(nodes[1][1], 0.0);
    EXPECT_EQ(nodes[2][0], 1.0);
    EXPECT_EQ(nodes[3][1], 1.0 / ny);
    EXPECT_EQ(nodes[9][1], 1.0);
}

// Test that
TEST_F(GridTest, FacesOfNodeCartGrid2d)
{
    EXPECT_EQ(grid_2d->num_faces(), 17);

    // Expected faces of selected nodes.
    std::map<int, std::vector<int>> expected_faces_of_node = {
        {0, {0, 9}},     {1, {1, 9, 10}}, {2, {2, 10}},      {3, {0, 3, 11}}, {4, {1, 4, 11, 12}},
        {5, {2, 5, 12}}, {9, {6, 15}},    {10, {7, 15, 16}}, {11, {8, 16}}};

    // Loop over the keys of the map, fetch the face nodes of the key and compare with the
    // expected values.
    for (auto const& [node, expected_faces] : expected_faces_of_node)
    {
        std::vector<int> face_nodes = grid_2d->faces_of_node(node);
        std::sort(face_nodes.begin(), face_nodes.end());
        EXPECT_EQ(face_nodes.size(), expected_faces.size());
        for (size_t i = 0; i < expected_faces.size(); ++i)
        {
            EXPECT_EQ(face_nodes[i], expected_faces[i]);
        }
    }
}

TEST_F(GridTest, NodesOfFaceCartGrid2d)
{
    // Expected nodes of selected faces.
    std::map<int, std::vector<int>> expected_nodes_of_face = {
        {0, {0, 3}},  {1, {1, 4}}, {2, {2, 5}},  {3, {3, 6}},  {5, {5, 8}},   {7, {7, 10}},
        {8, {8, 11}}, {9, {0, 1}}, {10, {1, 2}}, {11, {3, 4}}, {16, {10, 11}}};

    // Loop over the keys of the map, fetch the face nodes of the key and compare with the
    // expected values.
    for (auto const& [face, expected_nodes] : expected_nodes_of_face)
    {
        std::vector<int> face_nodes = grid_2d->nodes_of_face(face);
        std::sort(face_nodes.begin(), face_nodes.end());
        EXPECT_EQ(face_nodes.size(), expected_nodes.size());
        for (size_t i = 0; i < expected_nodes.size(); ++i)
        {
            EXPECT_EQ(face_nodes[i], expected_nodes[i]);
        }
    }
}