#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../include/compressed_storage.h"

namespace py = pybind11;

PYBIND11_MODULE(compressed_storage, m)
{
    py::class_<CompressedDataStorage<double>>(m, "CompressedDataStorage")
        .def(py::init<int, int>())  // Bind the empty matrix constructor
        .def(py::init<int, int, const std::vector<int>&, const std::vector<int>&,
                      const std::vector<double>&>())  // Bind the constructor with data
        .def("num_rows", &CompressedDataStorage<double>::num_rows)
        .def("num_cols", &CompressedDataStorage<double>::num_cols)
        .def("cols_in_row", &CompressedDataStorage<double>::cols_in_row)
        .def("rows_in_col", &CompressedDataStorage<double>::rows_in_col)
        .def("values", &CompressedDataStorage<double>::values)
        .def("value", &CompressedDataStorage<double>::value);
}