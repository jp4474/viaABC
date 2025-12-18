#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "grid.hpp"

namespace py = pybind11;

PYBIND11_MODULE(spatial2D_cpp, m)
{
    // ---------------- Parameters ----------------
    py::class_<Parameters>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("alpha", &Parameters::alpha)
        .def_readwrite("beta",  &Parameters::beta)
        .def_readwrite("gamma", &Parameters::gamma)
        .def_readwrite("dt",    &Parameters::dt)
        .def_readwrite("t0",    &Parameters::t0)
        .def_readwrite("t_end", &Parameters::t_end);

    // ---------------- Grid ----------------
    py::class_<Grid>(m, "Grid")
        .def(py::init<
            const std::vector<std::vector<int>>&,
            const Parameters&
        >())

        .def("simulate", &Grid::simulate)

        .def("shape",
             [](const Grid& g) {
                 return py::make_tuple(g.getRows(), g.getCols());
             })

        // ZERO-COPY NumPy view
        .def("numpy",
            [](Grid& g) {
                return py::array_t<uint8_t>(
                    {g.getRows(), g.getCols()},
                    {sizeof(uint8_t) * g.getCols(), sizeof(uint8_t)},
                    g.raw().data(),
                    py::cast(&g)
                );
            });
}
