#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "grid.hpp"

namespace py = pybind11;

// ----------------------------------------------------
// Python Module
// ----------------------------------------------------
PYBIND11_MODULE(cellular_sim_2d, m) {

    // Expose simulation parameters
    py::class_<Parameters>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("alpha", &Parameters::alpha)
        .def_readwrite("beta",  &Parameters::beta)
        .def_readwrite("gamma", &Parameters::gamma)
        .def_readwrite("dt",    &Parameters::dt)
        .def_readwrite("t0",    &Parameters::t0)
        .def_readwrite("t_end", &Parameters::t_end);

    // Expose grid class
    py::class_<Grid>(m, "Grid")
        .def(py::init<const std::string &, const Parameters &>())
        .def("simulate", &Grid::simulate)
        .def("getGrid",  &Grid::getGrid)
        .def("getRows",  &Grid::getRows)
        .def("getCols",  &Grid::getCols);

    // Factory: numpy → C++ grid
    m.def("GridFromNumpy",
        [](py::array_t<int> arr, const Parameters &params) {

            py::buffer_info info = arr.request();
            if (info.ndim != 2)
                throw std::runtime_error("Expected 2D numpy array.");

            int rows = info.shape[0];
            int cols = info.shape[1];

            int* ptr = static_cast<int*>(info.ptr);

            std::vector<std::vector<int>> data(rows, std::vector<int>(cols));

            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    data[r][c] = ptr[r * cols + c];

            return Grid(data, params);
        }
    );
}
