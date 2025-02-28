#include <gtest/gtest.h>

#include "../../src/utils/compressed_storage.cpp"

// Test fixture for CompressedDataStorage
class CompressedDataStorageTest : public ::testing::Test
{
   protected:
    int row_ptr[4] = {0, 2, 4, 4};
    int col_idx[4] = {0, 1, 1, 2};
    double values[4] = {1.0, 2.0, 3.0, 4.0};
    CompressedDataStorage<double>* storage;

    void SetUp() override
    {
        storage = new CompressedDataStorage<double>(3, 3, row_ptr, col_idx, values);
    }

    void TearDown() override
    {
        delete storage;
    }
};

// Test the constructor with indices and values
TEST_F(CompressedDataStorageTest, ConstructorWithValues)
{
    EXPECT_EQ(storage->num_rows(), 3);
    EXPECT_EQ(storage->num_cols(), 3);
    EXPECT_EQ(storage->values()[0], 1.0);
    EXPECT_EQ(storage->values()[1], 2.0);
    EXPECT_EQ(storage->values()[2], 3.0);
    EXPECT_EQ(storage->values()[3], 4.0);
}

// Test the cols_in_row method
TEST_F(CompressedDataStorageTest, ColsInRow)
{
    const int* cols = storage->cols_in_row(1);
    EXPECT_EQ(cols[0], 1);
    EXPECT_EQ(cols[1], 2);
    delete[] cols;
}

// Test the rows_in_col method
TEST_F(CompressedDataStorageTest, RowsInCol)
{
    const int* rows = storage->rows_in_col(1);
    EXPECT_EQ(rows[0], 0);
    EXPECT_EQ(rows[1], 1);
    delete[] rows;
}

// Test the values_in_row method
TEST_F(CompressedDataStorageTest, ValuesInRow)
{
    const double* row_values = storage->values_in_row(1);
    EXPECT_EQ(row_values[0], 3.0);
    EXPECT_EQ(row_values[1], 4.0);
    delete[] row_values;
}

// Main function to run all tests
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}