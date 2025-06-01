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
    SecondOrderTensor tensor_2d;
    ScalarDiscretization discr_2d;
    std::map<int, BoundaryCondition> bc_map_2d;

    // Constructor. Create a dummy tensor and empty discretization.
    MPFA() : grid_2d(nullptr), tensor_2d(2, 1, {1.0}), discr_2d(), bc_map_2d() {}

    void SetUp() override
    {
        // Create a simple grid. Let it be 3x3
        std::vector<int> num_cells = {3, 3};
        std::vector<double> lengths = {3.0, 3.0};
        grid_2d = Grid::create_cartesian_grid(2, num_cells, lengths);
        grid_2d->compute_geometry();

        std::vector<double> k_xx(grid_2d->num_cells());
        std::iota(k_xx.begin(), k_xx.end(), 1.0);  // Fill with 1.0, 2.0, ..., num_cells

        tensor_2d = SecondOrderTensor(2, 9, k_xx);
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

        // Compute the discretization
        discr_2d = mpfa(*grid_2d, tensor_2d, bc_map_2d);
    }
};

TEST_F(MPFA, FluxValuesInternalFace2dIsotropic)
{
    // Distance between the cell centers in the two directions.
    const double dx = 1.0;
    const double dy = 1.0;

    const double t_5 = dy / (0.5 * dx / 4.0 + 0.5 * dx / 5.0);

    const int face_ind = 5;  // Internal face index for the test.
    const std::vector<int> cell_indices = {0, 1, 3, 4, 6, 7};
    const std::vector<double> values = {0.0, 0.0, t_5, -t_5, 0.0, 0.0};

    std::cout << "Checking flux values for internal face " << face_ind << "\n";

    // for (size_t i = 0; i < cell_indices.size(); ++i)
    // {
    //     // Check the flux values for the internal faces.
    //     EXPECT_NEAR(discr_2d.flux->value(face_ind, cell_indices[i]), values[i], 1e-6);
    // }
}