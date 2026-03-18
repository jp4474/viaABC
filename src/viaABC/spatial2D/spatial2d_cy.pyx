# distutils: language=c++
# cython: language_level=3

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t, uint8_t
from libc.stddef cimport size_t

cdef extern from "spatial2d_api.hpp":
    ctypedef struct spatial2d_params_t:
        double alpha
        double beta
        double gamma
        double dt
        double t0
        double t_end

    ctypedef void* spatial2d_handle_t

    spatial2d_handle_t spatial2d_create(const int32_t* initial, size_t rows, size_t cols,
                                        spatial2d_params_t p)
    void spatial2d_destroy(spatial2d_handle_t h)
    int spatial2d_simulate(spatial2d_handle_t h, uint8_t* out)
    size_t spatial2d_rows(spatial2d_handle_t h)
    size_t spatial2d_cols(spatial2d_handle_t h)

cdef class Spatial2DCore:
    cdef spatial2d_handle_t h

    def __cinit__(self,
                  np.ndarray[np.int32_t, ndim=2] initial,
                  double dt, double t0, double t_end):
        if not initial.flags["C_CONTIGUOUS"]:
            initial = np.ascontiguousarray(initial, dtype=np.int32)

        cdef spatial2d_params_t p
        p.alpha = 0.0; p.beta = 0.0; p.gamma = 0.0
        p.dt = dt; p.t0 = t0; p.t_end = t_end

        cdef size_t r = initial.shape[0]
        cdef size_t c = initial.shape[1]

        self.h = spatial2d_create(<const int32_t*>initial.data, r, c, p)
        if self.h == NULL:
            raise RuntimeError("spatial2d_create failed")

    def __dealloc__(self):
        if self.h != NULL:
            spatial2d_destroy(self.h)
            self.h = NULL

    def simulation(self, double alpha, double beta, double gamma, double dt, double t0, double t_end):
        cdef size_t r = spatial2d_rows(self.h)
        cdef size_t c = spatial2d_cols(self.h)

        cdef np.ndarray[np.uint8_t, ndim=2] out = np.empty((r, c), dtype=np.uint8)
        cdef int rc = spatial2d_simulate(self.h, <uint8_t*>out.data)
        if rc != 0:
            raise RuntimeError(f"spatial2d_simulate failed with code {rc}")
        return out
