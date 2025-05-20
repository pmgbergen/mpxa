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

    for (size_t i = 0; i < num_data; ++i)
    {
        EXPECT_EQ(tensor_2d.isotropic_data(i), data[i]);

        auto diag = tensor_2d.diagonal_data(i);
        auto full = tensor_2d.full_data(i);
        EXPECT_EQ(diag.size(), 3u);
        EXPECT_EQ(full.size(), 6u);
        for (size_t j = 0; j < diag.size(); ++j)
        {
            EXPECT_EQ(diag[j], data[i]);
            EXPECT_EQ(full[j], data[i]);
        }
        for (size_t j = 3; j < full.size(); ++j)
        {
            EXPECT_EQ(full[j], 0.0);
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

    for (size_t i = 0; i < num_data; ++i)
    {
        auto diagonal_data = tensor_2d.diagonal_data(i);
        auto full_data = tensor_2d.full_data(i);
        for (size_t j = 3; j < 6; ++j)
        {
            EXPECT_EQ(full_data[j], 0);
        }
        EXPECT_EQ(tensor_2d.isotropic_data(i), data_xx[i]);
        EXPECT_EQ(diagonal_data[0], data_xx[i]);
        EXPECT_EQ(diagonal_data[1], data_yy[i]);
        // The zz component is not set, so it should be equal to the xx component.
        EXPECT_EQ(diagonal_data[2], data_xx[i]);
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

    for (size_t i = 0; i < num_data; ++i)
    {
        auto diagonal_data = tensor_3d.diagonal_data(i);
        auto full_data = tensor_3d.full_data(i);
        EXPECT_EQ(diagonal_data[0], data_xx[i]);
        EXPECT_EQ(full_data[0], data_xx[i]);
        EXPECT_EQ(diagonal_data[1], data_yy[i]);
        EXPECT_EQ(full_data[1], data_yy[i]);
        EXPECT_EQ(diagonal_data[2], data_zz[i]);
        EXPECT_EQ(full_data[2], data_zz[i]);
        // Full data[3:5] should be zero.
        for (size_t j = 3; j < 5; j++)
        {
            EXPECT_EQ(full_data[j], 0);
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

    for (size_t i = 0; i < num_data; ++i)
    {
        auto full_data = tensor_2d.full_data(i);
        EXPECT_EQ(full_data[0], data_xx[i]);
        EXPECT_EQ(full_data[1], data_yy[i]);
        EXPECT_EQ(full_data[3], data_xy[i]);
        // The zz component is not set, so it should be equal to the xx component.
        EXPECT_EQ(full_data[2], data_xx[i]);
        // The off-diagonal data should be zero.
        for (size_t j = 4; j < 6; ++j)
        {
            EXPECT_EQ(full_data[j], 0);
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

        for (size_t i = 0; i < num_data; ++i)
    {
        auto full_data = tensor_3d.full_data(i);
        EXPECT_EQ(full_data[0], data_xx[i]);
        EXPECT_EQ(full_data[1], data_yy[i]);
        EXPECT_EQ(full_data[2], data_zz[i]);
        EXPECT_EQ(full_data[3], data_xy[i]);
        EXPECT_EQ(full_data[4], data_xz[i]);
        EXPECT_EQ(full_data[5], data_yz[i]);
    }
}