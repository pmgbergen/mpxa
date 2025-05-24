#include <gtest/gtest.h>

#include "../../src/multipoint_common.cpp"

class BasisFunctionComputationTest : public ::testing::Test
{
   protected:
    BasisConstructor* basis_constructor;

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

    std::vector<std::vector<double>> basis_functions =
        basis_constructor->compute_basis_functions(coords_2d);

    // Check the size of the output
    EXPECT_EQ(basis_functions.size(), dim + 1);  // Should match the number of input coordinates
    for (size_t i = 0; i < basis_functions.size(); ++i)
    {
        EXPECT_EQ(basis_functions[i].size(), dim);  // Should match the dimension
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

    std::vector<std::vector<double>> basis_functions =
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
