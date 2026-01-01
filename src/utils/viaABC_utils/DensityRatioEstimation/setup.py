import sys
import setuptools
from setuptools import setup, Extension
import pybind11

# Find eigen include directory (Assuming it's installed in standard location or via conda)
# If Eigen is not found, you might need to hardcode the path e.g. include_dirs=['/usr/include/eigen3']
# or use `conda install eigen` and `sys.prefix + '/include/eigen3'`

ext_modules = [
    Extension(
        "dre_cpp",
        ["dre.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            "/usr/include/eigen3",
            "/usr/local/include/eigen3"
        ],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3"],
    ),
]

setup(
    name="dre_cpp",
    version="0.0.1",
    author="Jun Won Park",
    ext_modules=ext_modules,
)