# distutils: language=c++
# cython: language_level=3

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t, uint8_t
from libc.stddef cimport size_t

from libcpp.vector cimport vector

cdef extern from "grid.hpp" namespace "":
    cdef enum CellState:
        STATE_R
        STATE_Y
        STATE_B
        STATE_X
        STATE_G
        STATE_H

    cdef cppclass Parameters:
        double alpha
        double beta
        double gamma
        double dt
        double t0
        double t_end

    cdef cppclass Grid:
        Grid(const vector[vector[int]]& initial, const Parameters& params) except +
        void simulate() except +
        size_t getRows() const
        size_t getCols() const
        const vector[uint8_t]& raw() const

cdef class GridCore:
    """
    Python-facing wrapper around C++ Grid.
    We keep the initial condition as vector<vector<int>> so we can rebuild Grid
    for each parameter set (alpha,beta,gamma,dt,t0,t_end).
    """
    cdef vector[vector[int]] _initial2d
    cdef size_t _rows
    cdef size_t _cols

    def __cinit__(self, np.ndarray[np.int32_t, ndim=2] initial):
        """
        initial: (H, W) np.int32 contiguous array
        """
        if initial is None:
            raise ValueError("initial cannot be None")
        if not initial.flags["C_CONTIGUOUS"]:
            initial = np.ascontiguousarray(initial, dtype=np.int32)

        self._rows = <size_t>initial.shape[0]
        self._cols = <size_t>initial.shape[1]

        # Build vector<vector<int>> once (STL conversion happens here)
        self._initial2d.resize(self._rows)

        cdef size_t i, j
        for i in range(self._rows):
            self._initial2d[i].resize(self._cols)
            for j in range(self._cols):
                self._initial2d[i][j] = <int>initial[i, j]

    def simulation(self,
                double alpha, double beta, double gamma,
                double dt, double t0, double t_end):
        """
        Returns:
        out: (H, W) np.uint8  (labels 0..5)
        """
        cdef Parameters p
        cdef Grid* g = NULL
        cdef size_t r = 0
        cdef size_t c = 0
        cdef const vector[uint8_t]* v = NULL
        cdef np.ndarray[np.uint8_t, ndim=2] out
        cdef size_t k

        p.alpha = alpha
        p.beta = beta
        p.gamma = gamma
        p.dt = dt
        p.t0 = t0
        p.t_end = t_end

        g = new Grid(self._initial2d, p)
        try:
            g.simulate()

            r = g.getRows()
            c = g.getCols()

            v = &g.raw()

            if v[0].size() != r * c:
                raise ValueError(f"raw() size mismatch: {v[0].size()} vs {r*c}")

            out = np.empty((r, c), dtype=np.uint8)
            for k in range(r * c):
                out.flat[k] = v[0][k]

            return out
        finally:
            if g != NULL:
                del g
