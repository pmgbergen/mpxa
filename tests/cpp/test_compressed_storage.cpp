#include <gtest/gtest.h>

#include "../../include/compressed_storage.h"

// Test fixture for CompressedDataStorage
class CompressedDataStorageTest : public ::testing::Test
{
   protected:
    std::vector<int> row_ptr = {0, 2, 4, 4};
    std::vector<int> col_idx = {0, 1, 1, 2};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
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
    auto cols = storage->cols_in_row(1);
    EXPECT_EQ(cols[0], 1);
    EXPECT_EQ(cols[1], 2);
}

// Test the rows_in_col method
TEST_F(CompressedDataStorageTest, RowsInCol)
{
    std::vector<int> rows = storage->rows_in_col(1);
    EXPECT_EQ(rows[0], 0);
    EXPECT_EQ(rows[1], 1);
}

// Test the values_in_row method
TEST_F(CompressedDataStorageTest, ValuesInRow)
{
    EXPECT_EQ(storage->value(1, 0), 0.0);
    EXPECT_EQ(storage->value(1, 1), 3.0);
    EXPECT_EQ(storage->value(1, 2), 4.0);
}

// Main function to run all tests
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}