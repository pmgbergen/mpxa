#include <pybind11/pybind11.h>

#include "../../include/tensor.h"

namespace py = pybind11;

// PYBIND11_MODULE(tensor, m)
void init_tensor(py::module_ &m)
{
    py::class_<SecondOrderTensor>(m, "SecondOrderTensor")
        .def(py::init<int, int, std::vector<double>>());
    // .def("some_method", &SecondOrderTensor::some_method);  // Example method
}
