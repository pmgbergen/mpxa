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
    // Bindings for CompressedDataStorage.
    // Note to self: This tells pybind11 to use std::shared_ptr for the CompressedDataStorage
    // class (and not the unique_ptr, which is the default). This is necessary because we
    // want to use CompressedStorage in the python-binding of the Grid class, with memory
    // shared between the python and c++ side.
    // TODO: Not sure if this actually makes sense, since we copy the data into a std::vector
    // (actually, I believe it makes no sence), but it will have to do for now.
    py::class_<CompressedDataStorage<double>, std::shared_ptr<CompressedDataStorage<double>>>(
        m, "CompressedDataStorageDouble")
        .def(py::init(
            [](int num_rows, int num_cols, py::array_t<int> indptr, py::array_t<int> indices,
               py::array_t<double> data)
            {
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

    py::class_<CompressedDataStorage<int>, std::shared_ptr<CompressedDataStorage<int>>>(
        m, "CompressedDataStorageInt")
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