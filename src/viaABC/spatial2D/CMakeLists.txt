cmake_minimum_required(VERSION 3.15)
project(spatial2D_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Build Python module
pybind11_add_module(spatial2D_cpp
    bindings.cpp
    grid.cpp
)

# Optional but recommended optimizations
target_compile_options(spatial2D_cpp PRIVATE -O3)
