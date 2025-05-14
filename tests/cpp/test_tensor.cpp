#include <gtest/gtest.h>

#include <vector>

#include "../../src/tensor.cpp"

TEST(TensorTest, IsotropicSecondOrderTensor)
{
    const int num_data = 4;
    const std::vector<double> data = {1.0, 2.0, 3.0, 4.0};

    SecondOrderTensor tensor_2d(2, num_data, data);
    EXPECT_TRUE(tensor_2d.is_isotropic());
    EXPECT_TRUE(tensor_2d.is_diagonal());

    const std::vector<double>& isotropic_data = tensor_2d.isotropic_data();
    const std::vector<const double*> diagonal_data = tensor_2d.diagonal_data();
    const std::vector<const double*> full_data = tensor_2d.full_data();

    for (size_t i = 0; i < num_data; ++i)
    {
        EXPECT_EQ(isotropic_data[i], data[i]);
        EXPECT_EQ(diagonal_data[0][i], data[i]);
        for (size_t j = 1; j < 3; ++j)
        {
            // Diagonal and full data (along the diagonal) should be equal to isotropic data.
            EXPECT_EQ(diagonal_data[j][i], data[i]);
            EXPECT_EQ(full_data[j][i], data[i]);
        }
        // Full data off the diagonal should be zero.
        for (size_t j = 3; j < 6; ++j)
        {
            EXPECT_EQ(full_data[j][i], 0);
        }
    }
}

// Test a non-isotropic tensor in 2d
TEST(TensorTest, NonIsotropicSecondOrderTensor2d)
{
    const int num_data = 4;
    const std::vector<double> data_xx = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> data_yy = {5.0, 6.0, 7.0, 8.0};

    SecondOrderTensor tensor_2d(2, num_data, data_xx);
    tensor_2d.with_kyy(data_yy);

    EXPECT_FALSE(tensor_2d.is_isotropic());
    EXPECT_TRUE(tensor_2d.is_diagonal());

    const std::vector<double>& isotropic_data = tensor_2d.isotropic_data();
    const std::vector<const double*> diagonal_data = tensor_2d.diagonal_data();
    const std::vector<const double*> full_data = tensor_2d.full_data();

    for (size_t i = 0; i < num_data; ++i)
    {
        for (size_t j = 3; j < 6; ++j)
        {
            EXPECT_EQ(full_data[j][i], 0);
        }
        EXPECT_EQ(isotropic_data[i], data_xx[i]);
        EXPECT_EQ(diagonal_data[0][i], data_xx[i]);
        EXPECT_EQ(diagonal_data[1][i], data_yy[i]);
        // The zz component is not set, so it should be equal to the xx component.
        EXPECT_EQ(diagonal_data[2][i], data_xx[i]);
    }
}

// Test a non-isotropic tensor in 3d
TEST(TensorTest, NonIsotropicSecondOrderTensor3d)
{
    const int num_data = 4;
    const std::vector<double> data_xx = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> data_yy = {5.0, 6.0, 7.0, 8.0};
    const std::vector<double> data_zz = {9.0, 10.0, 11.0, 12.0};

    SecondOrderTensor tensor_3d(3, num_data, data_xx);
    tensor_3d.with_kyy(data_yy);
    tensor_3d.with_kzz(data_zz);

    EXPECT_FALSE(tensor_3d.is_isotropic());
    EXPECT_TRUE(tensor_3d.is_diagonal());

    const std::vector<double>& isotropic_data = tensor_3d.isotropic_data();
    const std::vector<const double*> diagonal_data = tensor_3d.diagonal_data();
    const std::vector<const double*> full_data = tensor_3d.full_data();

    for (size_t i = 0; i < num_data; ++i)
    {
        EXPECT_EQ(diagonal_data[0][i], data_xx[i]);
        EXPECT_EQ(full_data[0][i], data_xx[i]);
        EXPECT_EQ(diagonal_data[1][i], data_yy[i]);
        EXPECT_EQ(full_data[1][i], data_yy[i]);
        EXPECT_EQ(diagonal_data[2][i], data_zz[i]);
        EXPECT_EQ(full_data[2][i], data_zz[i]);
        // Full data[3:5] should be zero.
        for (size_t j = 4; j < 5; j++)
        {
            EXPECT_EQ(full_data[j][i], 0);
        }
    }
}

// Test a non-diagonal tensor in 2d
TEST(TensorTest, NonDiagonalSecondOrderTensor2d)
{
    const int num_data = 4;
    const std::vector<double> data_xx = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> data_yy = {5.0, 6.0, 7.0, 8.0};
    const std::vector<double> data_xy = {9.0, 10.0, 11.0, 12.0};

    SecondOrderTensor tensor_2d(2, num_data, data_xx);
    tensor_2d.with_kyy(data_yy);
    tensor_2d.with_kxy(data_xy);

    EXPECT_FALSE(tensor_2d.is_isotropic());
    EXPECT_FALSE(tensor_2d.is_diagonal());

    const std::vector<const double*> full_data = tensor_2d.full_data();

    for (size_t i = 0; i < num_data; ++i)
    {
        EXPECT_EQ(full_data[0][i], data_xx[i]);
        EXPECT_EQ(full_data[1][i], data_yy[i]);
        EXPECT_EQ(full_data[3][i], data_xy[i]);
        // The zz component is not set, so it should be equal to the xx component.
        EXPECT_EQ(full_data[2][i], data_xx[i]);
        // The off-diagonal data should be zero.
        for (size_t j = 4; j < 6; ++j)
        {
            EXPECT_EQ(full_data[j][i], 0);
        }
    }
}

// Test a non-diagonal tensor in 3d
TEST(TensorTest, NonDiagonalSecondOrderTensor3d)
{
    const int num_data = 4;
    const std::vector<double> data_xx = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> data_yy = {5.0, 6.0, 7.0, 8.0};
    const std::vector<double> data_zz = {9.0, 10.0, 11.0, 12.0};
    const std::vector<double> data_xy = {13.0, 14.0, 15.0, 16.0};
    const std::vector<double> data_xz = {17.0, 18.0, 19.0, 20.0};
    const std::vector<double> data_yz = {21.0, 22.0, 23.0, 24.0};

    SecondOrderTensor tensor_3d(3, num_data, data_xx);
    tensor_3d.with_kyy(data_yy);
    tensor_3d.with_kzz(data_zz);
    tensor_3d.with_kxy(data_xy);
    tensor_3d.with_kxz(data_xz);
    tensor_3d.with_kyz(data_yz);

    EXPECT_FALSE(tensor_3d.is_isotropic());
    EXPECT_FALSE(tensor_3d.is_diagonal());

    const std::vector<const double*> full_data = tensor_3d.full_data();

    for (size_t i = 0; i < num_data; ++i)
    {
        EXPECT_EQ(full_data[0][i], data_xx[i]);
        EXPECT_EQ(full_data[1][i], data_yy[i]);
        EXPECT_EQ(full_data[2][i], data_zz[i]);
        EXPECT_EQ(full_data[3][i], data_xy[i]);
        EXPECT_EQ(full_data[4][i], data_xz[i]);
        EXPECT_EQ(full_data[5][i], data_yz[i]);
    }
}