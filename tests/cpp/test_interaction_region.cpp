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
                     const std::vector<int>& expected_main_cells_of_faces)
{
    std::vector<int> actual_faces = interaction_region.faces();
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
    EXPECT_EQ(actual_main_cells_of_faces, expected_main_cells_of_faces);
}

TEST_F(InteractionRegionTest, Region2DGridCornerNode)
{
    // Create an interaction region for the 2D grid.
    InteractionRegion interaction_region_2d(0, 1, *grid_2d);

    std::vector<int> expected_faces = {0, 6};
    std::vector<int> expected_cells = {0};
    std::map<int, std::vector<int>> expected_faces_of_cells = {{0, {0, 6}}};
    std::vector<int> expected_main_cells_of_faces = {0, 0};

    validate_region(interaction_region_2d, expected_faces, expected_cells, expected_faces_of_cells,
                    expected_main_cells_of_faces);
}