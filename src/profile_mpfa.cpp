#include <chrono>
#include <iostream>
#include <numeric>

#include "../include/discr.h"  // Include necessary headers
#include "../include/grid.h"
#include "../include/multipoint_common.h"
#include "../include/tensor.h"

int main()
{
    // Create a mock Grid object
    std::unique_ptr<Grid> grid;
    // Initialize the grid with test data (you'll need to implement this)
    // grid.initialize(...);
    const std::vector<int> num_cells_2d = {10, 10};
    const std::vector<double> lengths_2d = {2.0, 3.0};
    const std::vector<double> unit_lengths_2d = {1.0, 1.0};

    // For 3D Cartesian grids.
    std::unique_ptr<Grid> grid_3d;
    const std::vector<int> num_cells_3d = {30, 20, 30};

    // Create grids, compute geometry.
    if (false)  // Replace with actual condition to choose between 2D and 3D
    {
        grid = Grid::create_cartesian_grid(2, num_cells_2d, lengths_2d);
    }
    else
    {
        grid = Grid::create_cartesian_grid(3, num_cells_3d, {2.0, 2.0, 2.0});
    }
    grid->compute_geometry();

    std::vector<double> k_xx(grid->num_cells());
    std::iota(k_xx.begin(), k_xx.end(), 1.0);  // Fill with 1.0, 2.0, ..., num_cells
    SecondOrderTensor tensor = SecondOrderTensor(grid->dim(), grid->num_cells(), k_xx);

    // Create a mock boundary condition map
    std::vector<int> boundary_faces = grid->boundary_faces();

    std::map<int, BoundaryCondition> bc_map;
    for (int face_id : boundary_faces)
    {
        // For testing, assign Dirichlet to even faces and Neumann to odd faces
        if (face_id % 2 == 0)
        {
            bc_map[face_id] = BoundaryCondition::Dirichlet;
        }
        else
        {
            bc_map[face_id] = BoundaryCondition::Neumann;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    // Call the mpfa function
    try
    {
        for (int i = 0; i < 3; ++i)
        {
            ScalarDiscretization result = mpfa(*grid, tensor, bc_map);
            std::cout << "mpfa function executed successfully." << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error during mpfa execution: " << e.what() << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time for 3 runs: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}