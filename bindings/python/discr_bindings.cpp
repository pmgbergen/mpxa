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
        .def_readwrite("bound_flux", &ScalarDiscretization::bound_flux);

    m.def("tpfa", &tpfa, py::arg("grid"), py::arg("tensor"), py::arg("bc_map"));
}
