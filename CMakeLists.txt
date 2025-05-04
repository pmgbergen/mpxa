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

# Output the Python version being used
message(STATUS "Using Python version: ${Python3_VERSION}")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include directory: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python3_LIBRARIES}")

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
file(GLOB_RECURSE PYTHON_BINDINGS_FILES bindings/python/*_bindings.cpp)

# Extract the corresponding implementation files from src
set(SRC_IMPLEMENTATION_FILES)
foreach(BINDING_FILE ${PYTHON_BINDINGS_FILES})
    get_filename_component(FILE_NAME ${BINDING_FILE} NAME_WE)
    if(NOT FILE_NAME STREQUAL "mpxa_bindings")  # Exclude mpxa_bindings.cpp
        string(REPLACE "_bindings" "" IMPLEMENTATION_FILE_NAME ${FILE_NAME})
        list(APPEND SRC_IMPLEMENTATION_FILES src/${IMPLEMENTATION_FILE_NAME}.cpp)
    endif()
endforeach()

pybind11_add_module(${PYTHON_PACKAGE_NAME}
    MODULE
    bindings/python/mpxa_bindings.cpp
    ${SRC_IMPLEMENTATION_FILES}
)
# target_link_libraries(${PYTHON_PACKAGE_NAME} PRIVATE libmpxa)

# Bundle the Python bindings and implementation files into a single library.
# add_library(${PYTHON_PACKAGE_NAME} MODULE "bindings/python/mpxa_bindings.cpp" ${SRC_IMPLEMENTATION_FILES})

# Link pybind11 library to the Python bindings.
# target_link_libraries(${PYTHON_PACKAGE_NAME} PRIVATE pybind11::module)

# Directly specify the path to liblibmpxa.so for linking
# target_link_libraries(${PYTHON_PACKAGE_NAME} PRIVATE ${CMAKE_BINARY_DIR}/src/liblibmpxa.so)

# Ensure the output is placed in the correct directory for Python to find
set_target_properties(${PYTHON_PACKAGE_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/python
)

# Add a message to guide the user during development
message(STATUS "Python bindings will be available in: ${CMAKE_BINARY_DIR}/bindings/python")

add_test(NAME ${TEST_EXECUTABLE} COMMAND ${TEST_EXECUTABLE})