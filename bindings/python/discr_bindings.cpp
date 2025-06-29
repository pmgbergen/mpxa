#include <pybind11/pybind11.h>
// #include <pybind11/smart_holder.h>
#include <pybind11/stl.h>

#include "../../include/discr.h"

void init_discr(py::module_ &m)
{
    // Bindings for ScalarDiscretization.
    py::class_<ScalarDiscretization>(m, "ScalarDiscretization")
        .def(py::init<>())
        .def_readwrite("flux", &ScalarDiscretization::flux)
        .def_readwrite("bound_flux", &ScalarDiscretization::bound_flux)
        .def_readwrite("vector_source", &ScalarDiscretization::vector_source)
        .def_readwrite("bound_vector_source", &ScalarDiscretization::bound_vector_source)
        .def_readwrite("bound_pressure_cell", &ScalarDiscretization::bound_pressure_cell)
        .def_readwrite("bound_pressure_face", &ScalarDiscretization::bound_pressure_face);

    // Bindings for BoundaryCondition enum.
    py::enum_<BoundaryCondition>(m, "BoundaryCondition")
        .value("Dirichlet", BoundaryCondition::Dirichlet)
        .value("Neumann", BoundaryCondition::Neumann)
        .value("Robin", BoundaryCondition::Robin)
        .export_values();

    // Bindings for tpfa function.
    m.def("tpfa", &tpfa, py::arg("grid"), py::arg("tensor"), py::arg("bc_map"));
    m.def("mpfa", &mpfa, py::arg("grid"), py::arg("tensor"), py::arg("bc_map"));
}
