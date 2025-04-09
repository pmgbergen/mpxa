#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../include/grid.h"  // Include only the header file

namespace py = pybind11;

PYBIND11_MODULE(grid, m)
{
    py::class_<Grid>(m, "Grid")
        .def(py::init<int, std::vector<std::vector<double>>, CompressedDataStorage<int>*,
                      CompressedDataStorage<int>*>())
        .def("dim", &Grid::dim)
        // .def("compute_geometry", &Grid::compute_geometry)
        // .def("create_cartesian_grid", &Grid::create_cartesian_grid)
        // .def("boundary_faces", &Grid::boundary_faces)
        .def("num_nodes", &Grid::num_nodes);
    // .def("num_faces", &Grid::num_faces)
    // .def("num_cells", &Grid::num_cells)
    // .def("faces_of_node", &Grid::faces_of_node)
    // .def("nodes_of_face", &Grid::nodes_of_face)
    // .def("cells_of_face", &Grid::cells_of_face)
    // .def("faces_of_cell", &Grid::faces_of_cell)
    // .def("sign_of_face_cell", &Grid::sign_of_face_cell)
    // .def("nodes", &Grid::nodes)
    // .def("cell_centers", &Grid::cell_centers)
    // .def("cell_volumes", &Grid::cell_volumes)
    // .def("face_areas", &Grid::face_areas)
    // .def("face_normals", &Grid::face_normals)
    // .def("face_centers", &Grid::face_centers)
    // .def("cell_center", &Grid::cell_center)
    // .def("cell_volume", &Grid::cell_volume)
    // .def("face_area", &Grid::face_area)
    // .def("face_normal", &Grid::face_normal)
    // .def("face_center", &Grid::face_center)
    // .def("set_cell_volumes", &Grid::set_cell_volumes)
    // .def("set_face_areas", &Grid::set_face_areas)
    // .def("set_face_normals", &Grid::set_face_normals)
    // .def("set_face_centers", &Grid::set_face_centers)
    // .def("set_cell_centers", &Grid::set_cell_centers);
}
