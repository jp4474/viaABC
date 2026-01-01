from setuptools import setup, Extension
import pybind11
import sys

ext_modules = [
    Extension(
        "particle_weight_cpp",
        ["particle_weight.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            "/usr/include/eigen3",
            "/usr/local/include/eigen3",
        ],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-std=c++11",
        ],
    ),
]

setup(
    name="particle_weight_cpp",
    version="0.0.1",
    author="Jun Won Park",
    ext_modules=ext_modules,
)
