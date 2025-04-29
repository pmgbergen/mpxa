#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "../../include/compressed_storage.h"

namespace py = pybind11;

// PYBIND11_MAKE_OPAQUE(std::vector<int>)
// PYBIND11_MAKE_OPAQUE(std::vector<double>)

void init_compressed_storage(py::module_& m)
{
    // py::bind_vector<std::vector<int>>(m, "VectorInt");
    // py::bind_vector<std::vector<double>>(m, "VectorDouble");

    py::class_<CompressedDataStorage<double>>(m, "CompressedDataStorageDouble")
        .def(py::init(
            [](int num_rows, int num_cols, py::array_t<int> indptr, py::array_t<int> indices,
               py::array_t<double> data)
            {
                // Get const pointers to the numpy array data
                const int* indptr_ptr = indptr.data();
                const int* indices_ptr = indices.data();
                const double* data_ptr = data.data();

                // Convert numpy arrays to std::vector. Note to self: This creates a
                // copy of the data (which on the one hand is not ideal, but on the
                // other hand, it seems necessary if we want to use std::vector on the
                // c++ side).
                std::vector<int> indptr_vec(indptr.data(), indptr.data() + indptr.size());
                std::vector<int> indices_vec(indices.data(), indices.data() + indices.size());
                std::vector<double> data_vec(data.data(), data.data() + data.size());

                // Return a new instance of CompressedDataStorage
                return std::make_unique<CompressedDataStorage<double>>(
                    num_rows, num_cols, indptr_vec, indices_vec, data_vec);
            }))
        .def("num_rows", &CompressedDataStorage<double>::num_rows)
        .def("num_cols", &CompressedDataStorage<double>::num_cols)
        // .def("cols_in_row", &CompressedDataStorage<double>::cols_in_row)
        // .def("rows_in_col", &CompressedDataStorage<double>::rows_in_col)
        // TODO: Return as a numpy array instead of a list.
        .def("values", &CompressedDataStorage<double>::values)
        .def("value", &CompressedDataStorage<double>::value);

    py::class_<CompressedDataStorage<int>>(m, "CompressedDataStorageInt")
        .def(py::init(
            [](int num_rows, int num_cols, py::array_t<int> indptr, py::array_t<int> indices,
               py::array_t<int> data)
            {
                // Convert numpy arrays to std::vector
                std::vector<int> indptr_vec(indptr.data(), indptr.data() + indptr.size());
                std::vector<int> indices_vec(indices.data(), indices.data() + indices.size());
                std::vector<int> data_vec(data.data(), data.data() + data.size());

                // Return a new instance of CompressedDataStorage
                return new CompressedDataStorage<int>(num_rows, num_cols, indptr_vec, indices_vec,
                                                      data_vec);
            }))
        .def("num_rows", &CompressedDataStorage<int>::num_rows)
        .def("num_cols", &CompressedDataStorage<int>::num_cols)
        // .def("cols_in_row", &CompressedDataStorage<int>::cols_in_row)
        // .def("rows_in_col", &CompressedDataStorage<int>::rows_in_col)
        .def("values", &CompressedDataStorage<int>::values)
        .def("value", &CompressedDataStorage<int>::value);
}