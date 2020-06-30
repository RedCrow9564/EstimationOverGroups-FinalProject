# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
"""
vectorized_actions.pyx - vectorized algorithms module
======================================================
This module contains methods for are implemented in Cython for Run-time improvements
"""
import numpy as np
cimport numpy as np
from numpy import kron
from scipy.linalg import eigh
from libc.math cimport sqrt
from Infrastructure.utils import Matrix

ctypedef fused double_or_complex:
    complex
    double

def extract_diagonals(double_or_complex[::1, :, :] g, const int signal_length) -> Matrix:
    cdef Py_ssize_t i, j
    cdef double[:] eigenvalue
    cdef double first_scaled_eigenvalue
    cdef double_or_complex[:, :] eigenvector
    cdef np.ndarray[double_or_complex, ndim=2] diagonals = np.empty((signal_length - 1, signal_length),
                                                                    dtype=g.base.dtype, order="F")

    for i in range(signal_length - 1):
        eigenvalue, eigenvector = eigh(
            g[i], overwrite_a = True, overwrite_b = True, check_finite = False,
            eigvals=(signal_length - 1, signal_length - 1))
        first_scaled_eigenvalue = sqrt(eigenvalue[0])
        for j in range(signal_length):
            diagonals[i, j] = first_scaled_eigenvalue * eigenvector[j, 0]

    return diagonals


def vectorized_kron(double_or_complex[::1, :, :] a) -> Matrix:
    cdef Py_ssize_t rows = a.shape[1]
    cdef Py_ssize_t cols = a.shape[2]
    cdef Py_ssize_t rows_sqr = rows * rows
    cdef Py_ssize_t cols_sqr = cols * cols
    cdef np.ndarray[double_or_complex, ndim=3] results = np.empty((rows, rows_sqr, cols_sqr),
                                                                  dtype=a.base.dtype, order="F")
    cdef Py_ssize_t i = 0

    if double_or_complex is complex:
        for i in range(rows - 1):
            results[i] = kron(a[i], np.conj(a[i + 1]))
        results[rows - 1] = kron(a[rows - 1], np.conj(a[0]))
    else:
        for i in range(rows - 1):
            results[i] = kron(a[i], a[i + 1])
        results[rows - 1] = kron(a[rows - 1], a[0])
    return results

def construct_estimator(double_or_complex[::1, :] diagonals, const double[::1] power_spectrum,
                        const int signal_length, const float noise_power) -> Matrix:
    cdef np.ndarray[double_or_complex, ndim=2] estimator = np.empty((signal_length, signal_length),
                                                                    dtype=diagonals.base.dtype)
    cdef Py_ssize_t i, j, column_index
    for i in range(signal_length):
        estimator[i, i] = <double_or_complex>(power_spectrum[i] - noise_power)

    for i in range(1, signal_length):
        for j in range(signal_length):
            column_index = <Py_ssize_t>((i + j) % signal_length)
            estimator[j, column_index] = diagonals[i - 1, j]

    return estimator
