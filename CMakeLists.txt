cmake_minimum_required(VERSION 3.15)
project(MPXA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add pybind11
# find_package(pybind11 REQUIRED)

# If pybind11 is installed via pip, you can use the following:
# execute_process(
#     COMMAND python3 -m pybind11 --cmakedir
#     OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
find_package(pybind11 REQUIRED PATHS ${PYBIND11_CMAKE_DIR})
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Build shared libraries. Not sure if this is the best option in the long term, but stay
# with it for now.
set(BUILD_SHARED_LIBS ON)

# Define the library name variable
set(LIBRARY_NAME libmpxa)
# Define the test executable name variable
set(TEST_EXECUTABLE test_mpxa)

# Set (empty) list of library dependencies
set(LIBRARY_DEPENDENCIES)

# Define the Python package name
set(PYTHON_PACKAGE_NAME "mpxa")

# Configure the binding files
# configure_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/grid_bindings.cpp.in
#     ${CMAKE_CURRENT_BINARY_DIR}/bindings/python/grid_bindings.cpp
#     @ONLY
# )

# # Update the include paths and sources
# include_directories(${CMAKE_CURRENT_BINARY_DIR}/bindings/python)
# add_library(python_bindings ${CMAKE_CURRENT_BINARY_DIR}/bindings/python/grid_bindings.cpp)

# Add the Google Test framework
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add subdirectories for source and tests
add_subdirectory(src)
add_subdirectory(tests)

# Add a target for the Python bindings
add_library(${PYTHON_PACKAGE_NAME} MODULE bindings/python/grid_bindings.cpp src/grid.cpp)
target_link_libraries(${PYTHON_PACKAGE_NAME} PRIVATE pybind11::module ${LIBRARY_DEPENDENCIES})

# Ensure the output is placed in the correct directory for Python to find
set_target_properties(${PYTHON_PACKAGE_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/python
)

# Add a message to guide the user during development
message(STATUS "Python bindings will be available in: ${CMAKE_BINARY_DIR}/bindings/python")

add_test(NAME ${TEST_EXECUTABLE} COMMAND ${TEST_EXECUTABLE})