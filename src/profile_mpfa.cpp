#include <chrono>
#include <iostream>
#include <numeric>

#include "../include/discr.h"  // Include necessary headers
#include "../include/grid.h"
#include "../include/multipoint_common.h"
#include "../include/tensor.h"

static std::unique_ptr<Grid> construct_bad_grid()
{
    // This constructs a 2D grid, tilted in a 3D domain. MPXA throws errows with it.
    // The grid is taken from a porepy integration test.
    return std::make_unique<Grid>(
        // dim
        2,
        // nodes (transposed! PP stores them in a transposed format.)
        std::vector<std::vector<double>>{{0.5, 0., 1.},
                                         {0.5, 0., 0.5},
                                         {0.5, 0., 0.5},
                                         {0.5, 0., 0.},
                                         {0.5, 0.5, 1.},
                                         {0.5, 0.5, 1.},
                                         {0.5, 0.5, 0.5},
                                         {0.5, 0.5, 0.5},
                                         {0.5, 0.5, 0.5},
                                         {0.5, 0.5, 0.5},
                                         {0.5, 0.5, 0.},
                                         {0.5, 0.5, 0.},
                                         {0.5, 1., 1.},
                                         {0.5, 1., 0.5},
                                         {0.5, 1., 0.5},
                                         {0.5, 1., 0.}},
        // cell_faces (in csr format! PP stores it in csc format.)
        std::make_shared<CompressedDataStorage<int>>(
            16, 4, std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
            std::vector<int>{0, 1, 1, 2, 3, 3, 0, 1, 2, 3, 2, 3, 0, 0, 1, 2},
            std::vector<int>{-1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1}),
        // faces_nodes (in csr format! PP stores it in csc format.)
        std::make_shared<CompressedDataStorage<int>>(
            16, 16, std::vector<int>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32},
            std::vector<int>{0, 6,  6, 13, 1, 7,  2, 7, 0, 12, 3,  8,  12, 13, 1, 14,
                             8, 15, 4, 9,  2, 14, 5, 9, 3, 10, 10, 15, 4,  11, 5, 11},
            std::vector<int>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

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
    const std::vector<int> num_cells_3d = {30, 80, 30};
    // const std::vector<int> num_cells_3d = {480, 40, 30};


    // Create grids, compute geometry.
    int grid_case = 2;
    if (grid_case == 0)  // Replace with actual condition to choose between 2D and 3D
    {
        // 2D grid
        grid = Grid::create_cartesian_grid(2, num_cells_2d, lengths_2d);
    }
    else if (grid_case == 1)
    {
        // 3D grid
        grid = Grid::create_cartesian_grid(3, num_cells_3d, {2.0, 2.0, 2.0});
    }
    else if (grid_case == 2)
    {
        // 2D grid tilted in 3D domain
        grid = construct_bad_grid();
    }
    else if (grid_case == 3)
    {
        // 1D grid
        grid = Grid::create_cartesian_grid(1, {30}, {2.0});
    }
    grid->compute_geometry();

    std::vector<double> k_xx(grid->num_cells());
    std::vector<double> k_yy(grid->num_cells());
    std::vector<double> k_zz(grid->num_cells());
    std::vector<double> k_xy(grid->num_cells());
    std::vector<double> k_xz(grid->num_cells());
    std::vector<double> k_yz(grid->num_cells());
    std::iota(k_xx.begin(), k_xx.end(), 1.0);  // Fill with 1.0, 2.0, ..., num_cells
    for (auto i{0}; i < grid->num_cells(); ++i)
    {
        k_yy[i] = k_xx[i] * 2;
        k_zz[i] = k_xx[i] * 3;
        k_xy[i] = k_xx[i] * -1;
        k_xz[i] = k_xx[i] * -0.5;
        k_yz[i] = k_xx[i] * -0.3;
    }

    SecondOrderTensor tensor = SecondOrderTensor(grid->dim(), grid->num_cells(), k_xx)
                                   .with_kyy(k_yy)
                                   .with_kzz(k_zz)
                                   .with_kxy(k_xy)
                                   .with_kxz(k_xz)
                                   .with_kyz(k_yz);

    bool anisotropic_tensor = false;
    if (anisotropic_tensor)
    {
        std::vector<double> k_yy(grid->num_cells());
        std::iota(k_yy.begin(), k_yy.end(), 2.0);
        tensor.with_kyy(k_yy);
    }

    // Create a mock boundary condition map
    std::vector<int> boundary_faces = grid->boundary_faces();

    std::unordered_map<int, BoundaryCondition> bc_map;
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

    int num_runs = 5;

    try
    {
        for (int i = 0; i < num_runs; ++i)
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
    std::cout << "Elapsed time for " << num_runs << " runs: " << elapsed.count() << " seconds."
              << std::endl;

    return 0;
}