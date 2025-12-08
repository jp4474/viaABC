cmake_minimum_required(VERSION 3.14)
project(cellular_sim_2d)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

pybind11_add_module(cellular_sim_2d
    bindings.cpp
    grid.cpp
)
