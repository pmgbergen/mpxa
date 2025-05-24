# Comments on usage (relative to what EK expects is common practice for CMake):
# 1. Python bindings are created by running cmake and then make, rather than using a
#    python-driven build system. This may change in the future, but will stay this way
#    for now.
# 2. To support user-installed versions of python (e.g., via uv) the cmake system will
#    look for the PYTHON_EXECUTABLE environment variable, and if set use this to find
#    the python executable.
# 3. The cmake system will locate the pybind11 library using the python executable, thus
#    allowing for a user-installed version of pybind11 to be used.

cmake_minimum_required(VERSION 3.15)
project(MPXA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Check if the PYTHON_EXECUTABLE environment variable is set. If so, use it as the
# Python executable.
if(DEFINED ENV{PYTHON_EXECUTABLE})
set(Python3_EXECUTABLE $ENV{PYTHON_EXECUTABLE})
    message(STATUS "Using Python executable from environment variable: ${Python3_EXECUTABLE}")
else()
    # If not set, use the default Python executable.
    message(WARNING "PYTHON_EXECUTABLE environment variable is not set. Using default Python.")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Locate the pybind11 library that python will find (assuming that virtual environments
# are set up and activated correctly).
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Force the setup to use the pybind11 cmake directory found above.
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED PATHS ${PYBIND11_CMAKE_DIR} NO_DEFAULT_PATH)

# Output the Python version being used.
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

include_directories(${CMAKE_SOURCE_DIR}/external/eigen)

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

# Find the implementation files in src.
file(GLOB_RECURSE SRC_IMPLEMENTATION_FILES src/*.cpp)

# Compile all the bindings specified in the file mpxa_bindings.cpp. Compile with all the
# source files in the src directory.
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