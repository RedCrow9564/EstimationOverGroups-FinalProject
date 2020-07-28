# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
"""
vectorized_actions.pyx - vectorized algorithms module
======================================================
This module contains methods for are implemented in Cython for run-time improvements
"""
import numpy as np
cimport numpy as np
from numpy import kron
from scipy.linalg import eigh
from numpy.fft import fft, ifft
from libc.math cimport sqrt
from Infrastructure.utils import Matrix, ThreeDMatrix

# A type which represents either a double-precision real number (64 bit - np.float64) or a double-precision
# complex number (128 bit - np.complex128).
ctypedef fused double_or_complex:
    complex
    double

def extract_diagonals(double_or_complex[::1, :, :] matrices_arr, const int signal_length) -> Matrix:
    """
    The function calculates the eigenvector of each matrices_arr[i] which corresponds to the largest (in magnitude)
    eigenvalue. This eigenvector is scaled by the square-root of the matching eigenvalue and stored as the i-th row
    of the result matrix.

    Args:
        matrices_arr(ThreeDMatrix): The matrices list, where square matrices are stored along the first dimension of
            the array, i.e matrices_arr[i] is the i-th square matrix in the array.
        signal_length(const int): The length of each dimension of all square matrices.

    Returns:
        A square matrix such that its i-th row is the scaled leading eigenvector of matrices_arr[i].
    """
    cdef Py_ssize_t i, j
    cdef double[:] eigenvalue
    cdef double first_scaled_eigenvalue
    cdef double_or_complex[:, :] eigenvector
    cdef np.ndarray[double_or_complex, ndim=2] diagonals = np.empty((signal_length - 1, signal_length),
                                                                    dtype=matrices_arr.base.dtype, order="F")

    for i in range(signal_length - 1):
        eigenvalue, eigenvector = eigh(
            matrices_arr[i], overwrite_a=True, overwrite_b=True, check_finite=False,
            eigvals=(signal_length - 1, signal_length - 1))
        first_scaled_eigenvalue = sqrt(eigenvalue[0])
        for j in range(signal_length):
            diagonals[i, j] = first_scaled_eigenvalue * eigenvector[j, 0]

    return diagonals


def vectorized_kron(double_or_complex[::1, :, :] matrices_arr) -> ThreeDMatrix:
    """
    The function for performing Kronecker product between each two consecutive matrices in a large 3D matrix

    Args:
        matrices_arr(ThreeDMatrix): The matrices list, where matrices are stored along the first dimension of
            the array, i.e matrices_arr[i] is the i-th matrix in the array.

    Returns:
        A 3D array which stores all these Kronecker products along its first axis.
    """
    cdef Py_ssize_t rows = matrices_arr.shape[1]
    cdef Py_ssize_t cols = matrices_arr.shape[2]
    cdef Py_ssize_t rows_sqr = rows * rows
    cdef Py_ssize_t cols_sqr = cols * cols
    cdef np.ndarray[double_or_complex, ndim=3] results = np.empty((rows, rows_sqr, cols_sqr),
                                                                  dtype=matrices_arr.base.dtype, order="F")
    cdef Py_ssize_t i = 0

    if double_or_complex is complex:
        for i in range(rows - 1):
            results[i] = kron(np.conj(matrices_arr[i + 1]), matrices_arr[i])
        results[rows - 1] = kron(np.conj(matrices_arr[0]), matrices_arr[rows - 1])
    else:
        for i in range(rows - 1):
            results[i] = kron(matrices_arr[i + 1], matrices_arr[i])
        results[rows - 1] = kron(matrices_arr[0], matrices_arr[rows - 1])
    return results


def construct_estimator(double_or_complex[::1, :] diagonals, const double[::1] power_spectrum,
                        const int signal_length, const float noise_power) -> Matrix:
    """
    The function creates a signal_length times signal_length matrix such that the i-th row of the diagonals matrix
    is the i-th diagonal of the matrix (with wrapping) for all 1<i<signal_length. The mai diagonal of this matrix
    is set to the power-spectrum estimation, minus the estimated noise power.

    Args:
        diagonals(Matrix): A 2D square matrix of size signal_length times signal_length.
        power_spectrum(const double[::1]): A vector of size signal_length which estimates of the signal's
            power-spectrum.
        signal_length(const int): The length of each dimension of all square matrices.
        noise_power(const float): An estimation for the signal's noise power :math:`\sigma^{2}`.

    Returns:
        A 3D array which stores all these Kronecker products along its first axis.
    """
    cdef np.ndarray[double_or_complex, ndim=2] estimator = np.empty((signal_length, signal_length),
                                                                    dtype=diagonals.base.dtype)
    cdef Py_ssize_t i, j, column_index

    # Constructing the main diagonal of the estimator.
    for i in range(signal_length):
        estimator[i, i] = <double_or_complex>(power_spectrum[i] - noise_power)

    # Constructing the off-diagonal terms.
    for i in range(1, signal_length):
        for j in range(signal_length):
            column_index = <Py_ssize_t>((i + j) % signal_length)
            estimator[j, column_index] = diagonals[i - 1, j]

    return estimator


def change_to_fourier_basis(double_or_complex[:, ::1] mat) -> Matrix:
    cdef np.ndarray[complex, ndim=2] fourier_basis_mat = np.empty_like(mat, dtype=np.complex128)
    fourier_basis_mat = np.conj(fft(mat, axis=0, norm="ortho").T)
    fourier_basis_mat = np.conj(fft(fourier_basis_mat, axis=0, norm="ortho").T)
    return fourier_basis_mat


def change_from_fourier_basis(double_or_complex[:, ::1] mat) -> Matrix:
    cdef np.ndarray[complex, ndim=2] original_basis_mat = np.empty_like(mat, dtype=np.complex128)
    original_basis_mat = np.conj(ifft(mat, axis=0, norm="ortho").T)
    original_basis_mat = np.conj(ifft(original_basis_mat, axis=0, norm="ortho").T)
    return original_basis_mat
