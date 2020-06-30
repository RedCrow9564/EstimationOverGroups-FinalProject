# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
"""
polyspectra_estimation.pyx - spectrum estimations algorithms module
===================================================================
This module contains methods for estimating a signal's power spectrum and tri-spectrum
from a given matrix of its observations.
"""
import numpy as np
cimport numpy as np
from Infrastructure.utils import Matrix
from libc.math cimport ceil

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex z)

def estimate_power_spectrum(const complex[:, ::1] observations_fourier) -> Matrix:
    """
    The function for estimating a signal's power-spectrum from its observations.

    Args:
        observations_fourier(Matrix): The fourier coefficients of all observations.

    Returns:
        A vector :math:`P_{y}` which estimates the power-spectrum of the original signal.
    """
    return np.mean(np.power(np.abs(observations_fourier), 2), axis=0)

def estimate_trispectrum(const complex[:, ::1] observations_fourier):
    """
    The function for estimating a signal's tri-spectrum from its observations.

    Args:
        observations_fourier(Matrix): The fourier coefficients of all observations.

    Returns:
        A 3D array :math:`T_{y}` which estimates the tri-spectrum of the original signal.
    """
    cdef Py_ssize_t signal_length = observations_fourier.shape[1]
    cdef Py_ssize_t observations_num = observations_fourier.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=3] trispectrum = np.empty(
        (signal_length, signal_length, signal_length), dtype=np.complex128, order='F')
    trispectrum_estimation_v1(observations_fourier, signal_length, observations_num, trispectrum)
    return trispectrum

cdef inline void trispectrum_estimation_v1(const complex[:, ::1] observations_fourier,
                                           const Py_ssize_t signal_length, const Py_ssize_t observations_num,
                                           complex[::1, :, :] trispectrum):
    cdef const double complex[:] observation
    cdef double complex temp = 0
    cdef Py_ssize_t i, j, k, m, s

    for i in range(signal_length):
        for j in range(signal_length):
            s = <Py_ssize_t>(i - j) % signal_length
            for k in range(signal_length):
                for m in range(observations_num):
                    observation = observations_fourier[m]
                    temp += observation[i] * conj(observation[j]) * observation[k] * conj(observation[s])
                trispectrum[i, j, k] = temp / <double complex>observations_num
                s = (s + 1) % signal_length
                temp = 0

def estimate_trispectrum_v2(const complex[:, ::1] observations_fourier):
    """
    The function for estimating a signal's tri-spectrum from its observations.

    Args:
        observations_fourier(Matrix): The fourier coefficients of all observations.

    Returns:
        A 3D array :math:`T_{y}` which estimates the tri-spectrum of the original signal.
    """
    cdef Py_ssize_t signal_length = observations_fourier.shape[1]
    cdef Py_ssize_t observations_num = observations_fourier.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=3] trispectrum = np.empty(
        (signal_length, signal_length, signal_length), dtype=np.complex128, order='F')
    trispectrum_estimation_v2(observations_fourier, signal_length, observations_num, trispectrum)
    return trispectrum

cdef inline void trispectrum_estimation_v2(const complex[:, ::1] observations_fourier,
                                           const Py_ssize_t signal_length, const Py_ssize_t observations_num,
                                           complex[::1, :, :] trispectrum):
    #TODO: Improve efficiency of this method by applying symmetries
    cdef const double complex[:] observation
    cdef double complex temp = 0
    cdef Py_ssize_t i, j, k, m, s
    cdef Py_ssize_t half_axis = <Py_ssize_t>ceil(<double>signal_length/2)

    for i in range(signal_length):
        for j in range(signal_length):
            for k in range(i):  # Applying a symmetry of the trispectrum: T[x,y,z]=T[z,y,x]
                trispectrum[i, j, k] = trispectrum[k, j, i]
            s = <Py_ssize_t>(2 * i - j) % signal_length
            for k in range(i, signal_length):
                for m in range(observations_num):
                    observation = observations_fourier[m]
                    temp += observation[i] * conj(observation[j]) * observation[k] * conj(observation[s])
                trispectrum[i, j, k] = temp / <double complex>observations_num
                s = (s + 1) % signal_length
                temp = 0
