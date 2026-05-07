#include <algorithm>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// #include <pybind11/smart_holder.h>
#include <pybind11/stl.h>

#include "../../include/grid.h"  // Include only the header file

namespace py = pybind11;

namespace
{
std::vector<double> copy_vector(std::span<const double> values)
{
    return std::vector<double>(values.begin(), values.end());
}
}  // namespace

// PYBIND11_MODULE(mpxa, m)
void init_grid(py::module_& m)
{
    using keep_alive_1_2 = py::keep_alive<1, 2>;

    py::class_<Grid>(m, "Grid")
        .def(py::init([](int dim,
                         py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
                         std::shared_ptr<CompressedDataStorage<int>> cell_faces,
                         std::shared_ptr<CompressedDataStorage<int>> face_nodes)
                      {
                          py::buffer_info buf = nodes.request();
                          const double* ptr = static_cast<const double*>(buf.ptr);
                          // Copy nodes into an owned vector. The input may be a
                          // temporary C-contiguous cast of a Fortran-ordered array,
                          // so we cannot safely hold a non-owning span to it.
                          std::vector<double> nodes_vec(ptr, ptr + buf.size);
                          return std::make_unique<Grid>(
                              dim, std::move(nodes_vec),
                              std::move(cell_faces), std::move(face_nodes));
                      }),
             py::arg("dim"), py::arg("nodes"), py::arg("faces_of_cell"),
             py::arg("nodes_of_face"))
        .def("dim", &Grid::dim)
        .def("compute_geometry", &Grid::compute_geometry)
        .def("create_cartesian_grid", &Grid::create_cartesian_grid)
        .def("boundary_faces", &Grid::boundary_faces)
        .def("num_nodes", &Grid::num_nodes)
        .def("num_faces", &Grid::num_faces)
        .def("num_cells", &Grid::num_cells)
        .def("faces_of_node", [](const Grid& g, int node) {
            auto s = g.faces_of_node(node);
            return std::vector<int>(s.begin(), s.end());
        })
        .def("nodes_of_face", &Grid::nodes_of_face)
        .def("cells_of_face", [](const Grid& g, int node) {
            auto s = g.cells_of_face(node);
            return std::vector<int>(s.begin(), s.end());
        })
        .def("faces_of_cell", &Grid::faces_of_cell)
        .def("sign_of_face_cell", &Grid::sign_of_face_cell)
        .def("nodes", [](const Grid& g) {
            auto span = g.nodes();
            int n = g.num_nodes();
            return py::array_t<double>({3, n}, span.data());
        })
        .def("cell_centers",
             [](const Grid& g) {
                 auto span = g.cell_centers();
                 int n = g.num_cells();
                 return py::array_t<double>({3, n}, span.data());
             })
        .def("cell_volumes", [](const Grid& g) { return copy_vector(g.cell_volumes()); })
        .def("face_areas", [](const Grid& g) { return copy_vector(g.face_areas()); })
        .def("face_normals",
             [](const Grid& g) {
                 auto span = g.face_normals();
                 int n = g.num_faces();
                 return py::array_t<double>({3, n}, span.data());
             })
        .def("face_centers",
             [](const Grid& g) {
                 auto span = g.face_centers();
                 int n = g.num_faces();
                 return py::array_t<double>({3, n}, span.data());
             })
        .def("node", &Grid::node)
        .def("cell_center", &Grid::cell_center)
        .def("cell_volume", &Grid::cell_volume)
        .def("face_area", &Grid::face_area)
        .def("face_normal", &Grid::face_normal)
        .def("face_center", &Grid::face_center)
        .def("set_cell_volumes",
             [](Grid& g, py::array_t<double, py::array::c_style | py::array::forcecast> arr)
             {
                 py::buffer_info buf = arr.request();
                 g.set_cell_volumes(static_cast<const double*>(buf.ptr), buf.size);
             },
             keep_alive_1_2{})
        .def("set_face_areas",
             [](Grid& g, py::array_t<double, py::array::c_style | py::array::forcecast> arr)
             {
                 py::buffer_info buf = arr.request();
                 g.set_face_areas(static_cast<const double*>(buf.ptr), buf.size);
             },
             keep_alive_1_2{})
        .def("set_face_normals",
             [](Grid& g, py::array_t<double, py::array::c_style | py::array::forcecast> arr)
             {
                 py::buffer_info buf = arr.request();
                 g.set_face_normals(static_cast<const double*>(buf.ptr), buf.size);
             },
             keep_alive_1_2{})
        .def("set_face_centers",
             [](Grid& g, py::array_t<double, py::array::c_style | py::array::forcecast> arr)
             {
                 py::buffer_info buf = arr.request();
                 g.set_face_centers(static_cast<const double*>(buf.ptr), buf.size);
             },
             keep_alive_1_2{})
        .def("set_cell_centers",
             [](Grid& g, py::array_t<double, py::array::c_style | py::array::forcecast> arr)
             {
                 py::buffer_info buf = arr.request();
                 g.set_cell_centers(static_cast<const double*>(buf.ptr), buf.size);
             },
             keep_alive_1_2{});
}
