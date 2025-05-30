#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "../../include/multipoint_common.h"

class InteractionRegionTest : public ::testing::Test
{
   protected:
    // For 2D Cartesian grids.
    std::unique_ptr<Grid> grid_2d;
    std::unique_ptr<Grid> grid_3d;

    void SetUp() override
    {
        // Create grids. An interaction region considers topology only, hence it should
        // work without setting up the geometry.
        grid_2d = Grid::create_cartesian_grid(2, {2, 2}, {2.0, 3.0});
        grid_3d = Grid::create_cartesian_grid(3, {2, 2, 2}, {2.0, 2.0, 2.0});
    }
};

bool sort_and_compare(const std::vector<int>& actual, const std::vector<int>& expected)
{
    std::vector<int> sorted_actual = actual;
    std::sort(sorted_actual.begin(), sorted_actual.end());
    if (sorted_actual == expected)
    {
        return true;
    }
    else
    {
        std::cerr << "Expected: ";
        for (const auto& val : expected)
        {
            std::cerr << val << " ";
        }
        std::cerr << "\nActual: ";
        for (const auto& val : sorted_actual)
        {
            std::cerr << val << " ";
        }
        std::cerr << std::endl;
        return false;
    }
}

void validate_region(const InteractionRegion& interaction_region,
                     const std::vector<int>& expected_faces, const std::vector<int>& expected_cells,
                     const std::map<int, std::vector<int>>& expected_faces_of_cells,
                     const std::vector<std::vector<int>>& expected_main_cells_of_faces)
{
    // Extract face indices from the map
    std::vector<int> actual_faces;
    for (const auto& kv : interaction_region.faces()) actual_faces.push_back(kv.first);

    std::vector<int> actual_cells = interaction_region.cells();
    std::map<int, std::vector<int>> actual_faces_of_cells = interaction_region.faces_of_cells();
    std::vector<int> actual_main_cells_of_faces = interaction_region.main_cell_of_faces();

    EXPECT_TRUE(sort_and_compare(actual_faces, expected_faces));
    EXPECT_TRUE(sort_and_compare(actual_cells, expected_cells));
    EXPECT_EQ(actual_faces_of_cells.size(), expected_faces_of_cells.size());
    for (const auto& [cell, faces] : expected_faces_of_cells)
    {
        EXPECT_TRUE(sort_and_compare(actual_faces_of_cells[cell], faces));
    }
    // For the main cells of faces, we can only check that one of the cells is the main cell.
    for (size_t i = 0; i < actual_main_cells_of_faces.size(); ++i)
    {
        bool found = false;
        for (const auto& expected_cell : expected_main_cells_of_faces[i])
        {
            if (std::find(actual_main_cells_of_faces.begin(), actual_main_cells_of_faces.end(),
                          expected_cell) != actual_main_cells_of_faces.end())
            {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Main cell " << actual_main_cells_of_faces[i] << " of face "
                           << actual_faces[i] << " does not match expected cells.";
    }
}

TEST_F(InteractionRegionTest, Region2DGridCornerNode)
{
    // Create an interaction region for the 2D grid.
    InteractionRegion interaction_region_2d(0, 1, *grid_2d);

    std::vector<int> expected_faces = {0, 6};
    std::vector<int> expected_cells = {0};
    std::map<int, std::vector<int>> expected_faces_of_cells = {{0, {0, 6}}};
    std::vector<std::vector<int>> expected_main_cells_of_faces = {{0}, {0}};

    validate_region(interaction_region_2d, expected_faces, expected_cells, expected_faces_of_cells,
                    expected_main_cells_of_faces);
}

TEST_F(InteractionRegionTest, Region2DGridEdgeNode)
{
    // Create an interaction region for the 2D grid.
    InteractionRegion interaction_region_2d(1, 1, *grid_2d);

    std::vector<int> expected_faces = {1, 6, 7};
    std::vector<int> expected_cells = {0, 1};
    std::map<int, std::vector<int>> expected_faces_of_cells = {{0, {1, 6}}, {1, {1, 7}}};
    std::vector<std::vector<int>> expected_main_cells_of_faces = {{0, 1}, {0}, {1}};

    validate_region(interaction_region_2d, expected_faces, expected_cells, expected_faces_of_cells,
                    expected_main_cells_of_faces);
}

TEST_F(InteractionRegionTest, Region2DGridInteriorNode)
{
    // Create an interaction region for the 2D grid.
    InteractionRegion interaction_region_2d(4, 1, *grid_2d);

    std::vector<int> expected_faces = {1, 4, 8, 9};
    std::vector<int> expected_cells = {0, 1, 2, 3};
    std::map<int, std::vector<int>> expected_faces_of_cells = {
        {0, {1, 8}}, {1, {1, 9}}, {2, {4, 8}}, {3, {4, 9}}};
    std::vector<std::vector<int>> expected_main_cells_of_faces = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};

    validate_region(interaction_region_2d, expected_faces, expected_cells, expected_faces_of_cells,
                    expected_main_cells_of_faces);
}

TEST_F(InteractionRegionTest, Region3DGridCornerNode)
{
    // Create an interaction region for the 3D grid.
    InteractionRegion interaction_region_3d(0, 1, *grid_3d);

    std::vector<int> expected_faces = {0, 12, 24};
    std::vector<int> expected_cells = {0};
    std::map<int, std::vector<int>> expected_faces_of_cells = {{0, {0, 12, 24}}};
    std::vector<std::vector<int>> expected_main_cells_of_faces = {{0}, {0}, {0}};

    validate_region(interaction_region_3d, expected_faces, expected_cells, expected_faces_of_cells,
                    expected_main_cells_of_faces);
}
TEST_F(InteractionRegionTest, Region3DGridEdgeNode)
{
    // Create an interaction region for the 3D grid.
    InteractionRegion interaction_region_3d(1, 1, *grid_3d);

    std::vector<int> expected_faces = {1, 12, 13, 24, 25};
    std::vector<int> expected_cells = {0, 1};
    std::map<int, std::vector<int>> expected_faces_of_cells = {{0, {1, 12, 24}}, {1, {1, 13, 25}}};
    std::vector<std::vector<int>> expected_main_cells_of_faces = {{0, 1}, {0}, {1}, {0}, {1}};

    validate_region(interaction_region_3d, expected_faces, expected_cells, expected_faces_of_cells,
                    expected_main_cells_of_faces);
}

TEST_F(InteractionRegionTest, Region3DGridSurfaceNode)
{
    // Create an interaction region for the 3D grid.
    InteractionRegion interaction_region_3d(10, 1, *grid_3d);

    std::vector<int> expected_faces = {1, 7, 12, 13, 18, 19, 28, 29};
    std::vector<int> expected_cells = {0, 1, 4, 5};
    std::map<int, std::vector<int>> expected_faces_of_cells = {
        {0, {1, 12, 28}}, {1, {1, 13, 29}}, {4, {7, 18, 28}}, {5, {7, 19, 29}}};
    std::vector<std::vector<int>> expected_main_cells_of_faces = {{0, 1}, {4, 5}, {0},    {1},
                                                                  {4},    {5},    {0, 4}, {1, 5}};

    validate_region(interaction_region_3d, expected_faces, expected_cells, expected_faces_of_cells,
                    expected_main_cells_of_faces);
}

TEST_F(InteractionRegionTest, Region3DGridInteriorNode)
{
    // Create an interaction region for the 3D grid.
    InteractionRegion interaction_region_3d(13, 1, *grid_3d);

    std::vector<int> expected_faces = {1, 4, 7, 10, 14, 15, 20, 21, 28, 29, 30, 31};
    std::vector<int> expected_cells = {0, 1, 2, 3, 4, 5, 6, 7};
    std::map<int, std::vector<int>> expected_faces_of_cells = {
        {0, {1, 14, 28}}, {1, {1, 15, 29}}, {2, {4, 14, 30}},  {3, {4, 15, 31}},
        {4, {7, 20, 28}}, {5, {7, 21, 29}}, {6, {10, 20, 30}}, {7, {10, 21, 31}}};
    std::vector<std::vector<int>> expected_main_cells_of_faces = {{0, 1}, {2, 3}, {4, 5}, {6, 7},
                                                                  {0, 2}, {1, 3}, {4, 6}, {5, 7},
                                                                  {0, 4}, {1, 5}, {2, 6}, {3, 7}};

    validate_region(interaction_region_3d, expected_faces, expected_cells, expected_faces_of_cells,
                    expected_main_cells_of_faces);
}
