cmake_minimum_required(VERSION 3.10)
project(MPXA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Build shared libraries. Not sure if this is the best option in the long term, but stay
# with it for now.
set(BUILD_SHARED_LIBS ON)

# Define the library name variable
set(LIBRARY_NAME mpxa)
# Define the test executable name variable
set(TEST_EXECUTABLE test_mpxa)

# Add the Google Test framework
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add subdirectories for source and tests
add_subdirectory(src)
add_subdirectory(tests)
