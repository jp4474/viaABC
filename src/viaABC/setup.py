from __future__ import annotations

from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import logging

HERE = Path(__file__).resolve().parent
logging.info(f"Setting up spatial2D extension module in {HERE}")

CPP_SOURCES = [
    str(HERE / "grid.cpp"),   
]

ext_modules = [
    Extension(
        name="viaABC.spatial2D._grid_core",         
        sources=[str(HERE / "grid_api.pyx"), *CPP_SOURCES],
        include_dirs=[
            np.get_include(),
            str(HERE),                         
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=[],
    )
]

setup(
    name="spatial2D",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)
