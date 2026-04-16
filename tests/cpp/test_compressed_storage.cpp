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
        storage = new CompressedDataStorage<double>(3, 3, std::move(row_ptr), std::move(col_idx),
                                                    std::move(values));
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

// Test that num_rows() and num_cols() are callable on a const reference
TEST_F(CompressedDataStorageTest, ConstAccessors)
{
    const CompressedDataStorage<double>& cstorage = *storage;
    EXPECT_EQ(cstorage.num_rows(), 3);
    EXPECT_EQ(cstorage.num_cols(), 3);
}

// Test the row_ptr() accessor
TEST_F(CompressedDataStorageTest, RowPtrAccessor)
{
    const auto& rp = storage->row_ptr();
    ASSERT_EQ(rp.size(), 4u);
    EXPECT_EQ(rp[0], 0);
    EXPECT_EQ(rp[1], 2);
    EXPECT_EQ(rp[2], 4);
    EXPECT_EQ(rp[3], 4);
}

// Test col_idx() and data() accessors
TEST_F(CompressedDataStorageTest, ColIdxAndDataAccessors)
{
    const auto& ci = storage->col_idx();
    ASSERT_EQ(ci.size(), 4u);
    EXPECT_EQ(ci[0], 0);
    EXPECT_EQ(ci[1], 1);

    const auto& d = storage->data();
    ASSERT_EQ(d.size(), 4u);
    EXPECT_DOUBLE_EQ(d[0], 1.0);
}

// Test constructor validation: row_ptr size mismatch
TEST(CompressedDataStorageValidation, ThrowsOnRowPtrSizeMismatch)
{
    std::vector<int> bad_row_ptr = {0, 2};  // should be size 4 for 3 rows
    std::vector<int> col_idx = {0, 1};
    std::vector<double> values = {1.0, 2.0};
    EXPECT_THROW(
        (CompressedDataStorage<double>(3, 3, std::move(bad_row_ptr), std::move(col_idx),
                                       std::move(values))),
        std::invalid_argument);
}

// Test constructor validation: col_idx and values size mismatch
TEST(CompressedDataStorageValidation, ThrowsOnColIdxValuesSizeMismatch)
{
    std::vector<int> row_ptr = {0, 2, 4, 4};
    std::vector<int> col_idx = {0, 1, 1, 2};
    std::vector<double> values = {1.0, 2.0};  // too short
    EXPECT_THROW(
        (CompressedDataStorage<double>(3, 3, std::move(row_ptr), std::move(col_idx),
                                       std::move(values))),
        std::invalid_argument);
}

// Test constructor validation: unsorted row_ptr
TEST(CompressedDataStorageValidation, ThrowsOnUnsortedRowPtr)
{
    std::vector<int> bad_row_ptr = {0, 4, 2, 4};  // row 1 has more entries than row 2 indicated
    std::vector<int> col_idx = {0, 1, 2, 3};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    EXPECT_THROW(
        (CompressedDataStorage<double>(3, 4, std::move(bad_row_ptr), std::move(col_idx),
                                       std::move(values))),
        std::invalid_argument);
}

// Test CSC construction and rows_in_col via the CSC fast path
TEST(CompressedDataStorageCSC, RowsInColViaCscPath)
{
    std::vector<int> row_ptr = {0, 2, 4, 4};
    std::vector<int> col_idx = {0, 1, 1, 2};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    CompressedDataStorage<double> csc_storage(3, 3, std::move(row_ptr), std::move(col_idx),
                                               std::move(values), /*construct_csc=*/true);
    std::vector<int> rows = csc_storage.rows_in_col(1);
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0], 0);
    EXPECT_EQ(rows[1], 1);
}

// Test CSC construction: column with no entries
TEST(CompressedDataStorageCSC, EmptyColumnViaCscPath)
{
    std::vector<int> row_ptr = {0, 1, 2, 2};
    std::vector<int> col_idx = {0, 1};
    std::vector<double> values = {1.0, 2.0};
    CompressedDataStorage<double> csc_storage(3, 3, std::move(row_ptr), std::move(col_idx),
                                               std::move(values), /*construct_csc=*/true);
    std::vector<int> rows = csc_storage.rows_in_col(2);
    EXPECT_TRUE(rows.empty());
}

// Main function to run all tests
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}