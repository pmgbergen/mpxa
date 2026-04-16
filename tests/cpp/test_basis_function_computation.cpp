#include <gtest/gtest.h>

#include "../../include/multipoint_common.h"

class BasisFunctionComputationTest : public ::testing::Test
{
   protected:
    BasisConstructor* basis_constructor = nullptr;

    void TearDown() override
    {
        delete basis_constructor;
    }
};

TEST_F(BasisFunctionComputationTest, ComputeBasisFunctions_2D)
{
    constexpr int dim = 2;
    basis_constructor = new BasisConstructor(dim);

    // For the 2D test, we ust need the first three coordinates
    std::vector<std::array<double, 3>> coords_2d = {
        {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {1.0, 3.0, 0.0}};

    std::vector<std::array<double, 3>> basis_functions =
        basis_constructor->compute_basis_functions(coords_2d);

    // Check the size of the output
    EXPECT_EQ(basis_functions.size(), dim + 1);  // Should match the number of input coordinates
    for (size_t i = 0; i < basis_functions.size(); ++i)
    {
        EXPECT_EQ(basis_functions[i].size(), 3);  // Should match the dimension
    }

    // Check specific values. First the function for the first coordinate.
    EXPECT_NEAR(basis_functions[0][0], -1.0 / 2, 1e-6);
    EXPECT_NEAR(basis_functions[0][1], -1.0 / 6, 1e-6);
    EXPECT_NEAR(basis_functions[0][2], 0.0,
                1e-6);  // The z-coordinate should not affect the 2D basis function.
    // Now the function for the second coordinate.
    EXPECT_NEAR(basis_functions[1][0], 1.0 / 2, 1e-6);
    EXPECT_NEAR(basis_functions[1][1], -1.0 / 6, 1e-6);
    EXPECT_NEAR(basis_functions[1][2], 0.0, 1e-6);
    // Finally the function for the third coordinate.
    EXPECT_NEAR(basis_functions[2][0], 0.0, 1e-6);
    EXPECT_NEAR(basis_functions[2][1], 1.0 / 3, 1e-6);  // Well done, copilot!
    EXPECT_NEAR(basis_functions[2][2], 0.0, 1e-6);
}

TEST_F(BasisFunctionComputationTest, ComputeBasisFunctions_3D)
{
    constexpr int dim = 3;
    basis_constructor = new BasisConstructor(dim);

    // For the 2D test, we ust need the first three coordinates
    std::vector<std::array<double, 3>> coords_3d = {
        {0.0, 0.0, 0.0}, {2.0, 0.0, 1.0}, {1.0, 3.0, 0.0}, {1.0, 1.0, 2.0}};

    std::vector<std::array<double, 3>> basis_functions =
        basis_constructor->compute_basis_functions(coords_3d);

    // Check the size of the output
    EXPECT_EQ(basis_functions.size(), dim + 1);  // Should match the number of input coordinates
    for (size_t i = 0; i < basis_functions.size(); ++i)
    {
        EXPECT_EQ(basis_functions[i].size(), dim);  // Should match the dimension
    }

    // Check specific values. First the function for the first coordinate.
    EXPECT_NEAR(basis_functions[0][0], -0.4, 1e-6);
    EXPECT_NEAR(basis_functions[0][1], -0.2, 1e-6);
    EXPECT_NEAR(basis_functions[0][2], -0.2, 1e-6);
    EXPECT_NEAR(basis_functions[1][0], 0.6, 1e-6);
    EXPECT_NEAR(basis_functions[1][1], -0.2, 1e-6);
    EXPECT_NEAR(basis_functions[1][2], -0.2, 1e-6);
    // the function for the third point.
    EXPECT_NEAR(basis_functions[2][0], 0.1, 1e-6);
    EXPECT_NEAR(basis_functions[2][1], 0.3, 1e-6);
    EXPECT_NEAR(basis_functions[2][2], -0.2, 1e-6);
    // the function for the fourth point.
    EXPECT_NEAR(basis_functions[3][0], -0.3, 1e-6);
    EXPECT_NEAR(basis_functions[3][1], 0.1, 1e-6);
    EXPECT_NEAR(basis_functions[3][2], 0.6, 1e-6);
}

TEST_F(BasisFunctionComputationTest, InvalidDimThrows)
{
    BasisConstructor bc1(0);
    BasisConstructor bc4(4);
    const std::vector<std::array<double, 3>> dummy = {{0, 0, 0}};
    EXPECT_THROW(bc1.compute_basis_functions(dummy), std::runtime_error);
    EXPECT_THROW(bc4.compute_basis_functions(dummy), std::runtime_error);
}

TEST_F(BasisFunctionComputationTest, NonZeroZCoordIn2DThrows)
{
    BasisConstructor bc(2);
    const std::vector<std::array<double, 3>> coords = {
        {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    EXPECT_THROW(bc.compute_basis_functions(coords), std::logic_error);
}

TEST_F(BasisFunctionComputationTest, DegenerateTriangleThrows)
{
    BasisConstructor bc(2);
    // Three collinear points — det = 0.
    const std::vector<std::array<double, 3>> coords = {
        {0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {2.0, 2.0, 0.0}};
    EXPECT_THROW(bc.compute_basis_functions(coords), std::logic_error);
}

// Verify partition-of-unity: sum of all basis-function gradients must be zero.
TEST_F(BasisFunctionComputationTest, PartitionOfUnity2D)
{
    BasisConstructor bc(2);
    const std::vector<std::array<double, 3>> coords = {
        {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {1.0, 3.0, 0.0}};
    auto bf = bc.compute_basis_functions(coords);
    double sum_x = 0.0, sum_y = 0.0;
    for (const auto& g : bf)
    {
        sum_x += g[0];
        sum_y += g[1];
    }
    EXPECT_NEAR(sum_x, 0.0, 1e-10);
    EXPECT_NEAR(sum_y, 0.0, 1e-10);
}

TEST_F(BasisFunctionComputationTest, PartitionOfUnity3D)
{
    BasisConstructor bc(3);
    const std::vector<std::array<double, 3>> coords = {
        {0.0, 0.0, 0.0}, {2.0, 0.0, 1.0}, {1.0, 3.0, 0.0}, {1.0, 1.0, 2.0}};
    auto bf = bc.compute_basis_functions(coords);
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    for (const auto& g : bf)
    {
        sum_x += g[0];
        sum_y += g[1];
        sum_z += g[2];
    }
    EXPECT_NEAR(sum_x, 0.0, 1e-10);
    EXPECT_NEAR(sum_y, 0.0, 1e-10);
    EXPECT_NEAR(sum_z, 0.0, 1e-10);
}
