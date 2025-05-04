#include <pybind11/pybind11.h>

#include "compressed_storage_bindings.cpp"  // Include the header or source file
#include "discr_bindings.cpp"               // Include the header or source file
#include "grid_bindings.cpp"                // Include the header or source file
#include "tensor_bindings.cpp"              // Include the header or source file

namespace py = pybind11;

PYBIND11_MODULE(mpxa, m)
{
    // m.doc() = "MPXA Python module";  // Optional module docstring

    // Initialize bindings for compressed_storage
    init_compressed_storage(m);
    // Initialize bindings for grid
    init_grid(m);
    // Initialize bindings for tensor
    init_tensor(m);
    // Bindings for the discretizations
    init_discr(m);
}