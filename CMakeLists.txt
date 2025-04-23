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

# Add the Google Test framework
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add the include directory globally to all targets
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add subdirectories for source and tests
add_subdirectory(src)
add_subdirectory(tests)

# Dynamically search for all binding files in bindings/python and their corresponding implementation files in src
file(GLOB_RECURSE PYTHON_BINDINGS_SOURCES bindings/python/*_bindings.cpp)

# Extract the corresponding implementation files from src
set(SRC_IMPLEMENTATION_FILES)
foreach(BINDING_FILE ${PYTHON_BINDINGS_SOURCES})
    get_filename_component(FILE_NAME ${BINDING_FILE} NAME_WE)
    string(REPLACE "_bindings" "" IMPLEMENTATION_FILE_NAME ${FILE_NAME})
    list(APPEND SRC_IMPLEMENTATION_FILES src/${IMPLEMENTATION_FILE_NAME}.cpp)
endforeach()

add_library(${PYTHON_PACKAGE_NAME} MODULE ${PYTHON_BINDINGS_SOURCES} ${SRC_IMPLEMENTATION_FILES})
target_link_libraries(${PYTHON_PACKAGE_NAME} PRIVATE pybind11::module ${LIBRARY_DEPENDENCIES})

# Link the libmpxa library to the Python bindings
target_link_libraries(${PYTHON_PACKAGE_NAME} PRIVATE pybind11::module ${LIBRARY_NAME})

# Directly specify the path to liblibmpxa.so for linking
target_link_libraries(${PYTHON_PACKAGE_NAME} PRIVATE ${CMAKE_BINARY_DIR}/src/liblibmpxa.so)

# Ensure the output is placed in the correct directory for Python to find
set_target_properties(${PYTHON_PACKAGE_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/python
)

# Add RPATH to ensure runtime linking to liblibmpxa.so
set_target_properties(${PYTHON_PACKAGE_NAME} PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/src"
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/src"
)

# Add a message to guide the user during development
message(STATUS "Python bindings will be available in: ${CMAKE_BINARY_DIR}/bindings/python")

add_test(NAME ${TEST_EXECUTABLE} COMMAND ${TEST_EXECUTABLE})