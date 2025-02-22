# Recursively find all .cpp and .h files in the src directory
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "*.cpp" "*.h")

# Add the library target
add_library(${LIBRARY_NAME} ${SRC_FILES})