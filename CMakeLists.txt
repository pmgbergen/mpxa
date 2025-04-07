cmake_minimum_required(VERSION 3.10)
project(MPXA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add pybind11
# find_package(pybind11 REQUIRED)

# If pybind11 is installed via pip, you can use the following:
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${PYBIND11_CMAKE_DIR})

# Build shared libraries. Not sure if this is the best option in the long term, but stay
# with it for now.
set(BUILD_SHARED_LIBS ON)

# Define the library name variable
set(LIBRARY_NAME mpxa)
# Define the test executable name variable
set(TEST_EXECUTABLE test_mpxa)

# Set (empty) list of library dependencies
set(LIBRARY_DEPENDENCIES)

# Add the Google Test framework
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add subdirectories for source and tests
add_subdirectory(src)
add_subdirectory(tests)


add_test(NAME ${TEST_EXECUTABLE} COMMAND ${TEST_EXECUTABLE})