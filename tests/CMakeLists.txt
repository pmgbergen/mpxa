# Recursively find all .cpp and .h files in the tests directory
file(GLOB_RECURSE TEST_FILES CONFIGURE_DEPENDS "*.cpp" "*.h")

# Add the test executable
add_executable(${TEST_EXECUTABLE} ${TEST_FILES})

# Link the test executable with Google Test and the library
target_link_libraries(${TEST_EXECUTABLE} ${GTEST_LIBRARIES})# pthread ${LIBRARY_NAME})

# Add the test to CTest
