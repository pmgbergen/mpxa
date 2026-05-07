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
    // Type aliases to work around C++ template argument parsing ambiguity when
    // using py::keep_alive<N, M> directly in .def() argument lists.
    // keep_alive<1, N>: used with py::init — keep argument N alive as long as self (pos 1) is alive.
    using keep_alive_1_3 = py::keep_alive<1, 3>;
    using keep_alive_1_4 = py::keep_alive<1, 4>;
    using keep_alive_1_5 = py::keep_alive<1, 5>;
    // keep_alive<0, N>: used with def_static — keep argument N alive as long as return value is alive.
    using keep_alive_0_3 = py::keep_alive<0, 3>;
    using keep_alive_0_4 = py::keep_alive<0, 4>;
    using keep_alive_0_5 = py::keep_alive<0, 5>;

    py::class_<CompressedDataStorage<double>, std::shared_ptr<CompressedDataStorage<double>>>(
        m, "CompressedDataStorageDouble")
        .def(py::init(
                 [](int num_rows, int num_cols,
                    py::array_t<int, py::array::c_style | py::array::forcecast> indptr,
                    py::array_t<int, py::array::c_style | py::array::forcecast> indices,
                    py::array_t<double, py::array::c_style | py::array::forcecast> data,
                    bool csc)
                 {
                     // Pass numpy array data pointers directly — no copy into std::vector.
                     // The numpy arrays are kept alive by py::keep_alive below.
                     py::buffer_info indptr_buf = indptr.request();
                     py::buffer_info indices_buf = indices.request();
                     py::buffer_info data_buf = data.request();

                     return std::make_shared<CompressedDataStorage<double>>(
                         num_rows, num_cols,
                         static_cast<const int*>(indptr_buf.ptr), indptr_buf.size,
                         static_cast<const int*>(indices_buf.ptr), indices_buf.size,
                         static_cast<const double*>(data_buf.ptr), data_buf.size, csc);
                 }),
             keep_alive_1_3{}, keep_alive_1_4{}, keep_alive_1_5{})
        .def("num_rows", &CompressedDataStorage<double>::num_rows)
        .def("num_cols", &CompressedDataStorage<double>::num_cols)
        // Return zero-copy numpy views backed by the C++ span.  Passing `self`
        // as the base object keeps the CompressedDataStorage Python wrapper alive
        // as long as the returned array is alive, so the span memory is valid.
        .def("row_ptr",
             [](py::object self)
             {
                 auto sp = self.cast<const CompressedDataStorage<double>&>().row_ptr();
                 return py::array_t<int>({(py::ssize_t)sp.size()},
                                        {(py::ssize_t)sizeof(int)}, sp.data(), self);
             })
        .def("col_idx",
             [](py::object self)
             {
                 auto sp = self.cast<const CompressedDataStorage<double>&>().col_idx();
                 return py::array_t<int>({(py::ssize_t)sp.size()},
                                        {(py::ssize_t)sizeof(int)}, sp.data(), self);
             })
        .def("data",
             [](py::object self)
             {
                 auto sp = self.cast<const CompressedDataStorage<double>&>().data();
                 return py::array_t<double>({(py::ssize_t)sp.size()},
                                           {(py::ssize_t)sizeof(double)}, sp.data(), self);
             })
        .def("values", &CompressedDataStorage<double>::values)
        .def("value", &CompressedDataStorage<double>::value)
        .def_static(
            "from_csc",
            [](int num_rows, int num_cols,
               py::array_t<int, py::array::c_style | py::array::forcecast> col_ptr,
               py::array_t<int, py::array::c_style | py::array::forcecast> row_idx,
               py::array_t<double, py::array::c_style | py::array::forcecast> values)
            {
                py::buffer_info col_ptr_buf = col_ptr.request();
                py::buffer_info row_idx_buf = row_idx.request();
                py::buffer_info values_buf = values.request();
                return CompressedDataStorage<double>::from_csc(
                    num_rows, num_cols,
                    static_cast<const int*>(col_ptr_buf.ptr), col_ptr_buf.size,
                    static_cast<const int*>(row_idx_buf.ptr), row_idx_buf.size,
                    static_cast<const double*>(values_buf.ptr), values_buf.size);
            },
            keep_alive_0_3{}, keep_alive_0_4{}, keep_alive_0_5{});

    py::class_<CompressedDataStorage<int>, std::shared_ptr<CompressedDataStorage<int>>>(
        m, "CompressedDataStorageInt")
        .def(py::init(
                 [](int num_rows, int num_cols,
                    py::array_t<int, py::array::c_style | py::array::forcecast> indptr,
                    py::array_t<int, py::array::c_style | py::array::forcecast> indices,
                    py::array_t<int, py::array::c_style | py::array::forcecast> data,
                    bool csc)
                 {
                     // Pass numpy array data pointers directly — no copy into std::vector.
                     // The numpy arrays are kept alive by py::keep_alive below.
                     py::buffer_info indptr_buf = indptr.request();
                     py::buffer_info indices_buf = indices.request();
                     py::buffer_info data_buf = data.request();

                     return std::make_shared<CompressedDataStorage<int>>(
                         num_rows, num_cols,
                         static_cast<const int*>(indptr_buf.ptr), indptr_buf.size,
                         static_cast<const int*>(indices_buf.ptr), indices_buf.size,
                         static_cast<const int*>(data_buf.ptr), data_buf.size, csc);
                 }),
             keep_alive_1_3{}, keep_alive_1_4{}, keep_alive_1_5{})
        .def("num_rows", &CompressedDataStorage<int>::num_rows)
        .def("num_cols", &CompressedDataStorage<int>::num_cols)
        // Return zero-copy numpy views backed by the C++ span.  Passing `self`
        // as the base object keeps the CompressedDataStorage Python wrapper alive
        // as long as the returned array is alive, so the span memory is valid.
        .def("row_ptr",
             [](py::object self)
             {
                 auto sp = self.cast<const CompressedDataStorage<int>&>().row_ptr();
                 return py::array_t<int>({(py::ssize_t)sp.size()},
                                        {(py::ssize_t)sizeof(int)}, sp.data(), self);
             })
        .def("col_idx",
             [](py::object self)
             {
                 auto sp = self.cast<const CompressedDataStorage<int>&>().col_idx();
                 return py::array_t<int>({(py::ssize_t)sp.size()},
                                        {(py::ssize_t)sizeof(int)}, sp.data(), self);
             })
        .def("data",
             [](py::object self)
             {
                 auto sp = self.cast<const CompressedDataStorage<int>&>().data();
                 return py::array_t<int>({(py::ssize_t)sp.size()},
                                        {(py::ssize_t)sizeof(int)}, sp.data(), self);
             })
        .def("values", &CompressedDataStorage<int>::values)
        .def("value", &CompressedDataStorage<int>::value)
        .def_static(
            "from_csc",
            [](int num_rows, int num_cols,
               py::array_t<int, py::array::c_style | py::array::forcecast> col_ptr,
               py::array_t<int, py::array::c_style | py::array::forcecast> row_idx,
               py::array_t<int, py::array::c_style | py::array::forcecast> values)
            {
                py::buffer_info col_ptr_buf = col_ptr.request();
                py::buffer_info row_idx_buf = row_idx.request();
                py::buffer_info values_buf = values.request();
                return CompressedDataStorage<int>::from_csc(
                    num_rows, num_cols,
                    static_cast<const int*>(col_ptr_buf.ptr), col_ptr_buf.size,
                    static_cast<const int*>(row_idx_buf.ptr), row_idx_buf.size,
                    static_cast<const int*>(values_buf.ptr), values_buf.size);
            },
            keep_alive_0_3{}, keep_alive_0_4{}, keep_alive_0_5{});
}