# -*- coding: utf-8 -*-
"""
test_optimization_step.py - tests for the optimization step components
======================================================================

This module contains the tests for the optimization step in Algorithm 1
in the paper.
"""
import unittest
from typing import Callable
import numpy as np
from scipy.linalg import eigh
from scipy.linalg import circulant
from numpy.random import Generator, PCG64
from Infrastructure.utils import Vector, Matrix, ThreeDMatrix
from Infrastructure import pyximportcpp; pyximportcpp.install(setup_args={"include_dirs": np.get_include()},
                                                              reload_support=True)
from covariance_estimation import create_optimization_objective
from data_generation import generate_covariance
from UnitTests.test_tri_spectrum_estimation import calc_exact_tri_spectrum


def find_diagonals(exact_covariance: Matrix) -> Matrix:
    """
    The function extracts the diagonals of the exact covariance matrix (in Fourier basis).
    Args:
        exact_covariance(Matrix): The exact covariance (square) matrix in Fourier basis.

    Returns:
        A square matrix such that its i-th row is the i-th diagonal of the input matrix.
    """
    signal_length: int = exact_covariance.shape[0]
    diags: Matrix = np.empty((signal_length - 1, signal_length), dtype=exact_covariance.dtype)

    for i in range(1, signal_length):
        diags[i - 1] = np.hstack((np.diagonal(exact_covariance, offset=i),
                                  np.diagonal(exact_covariance, offset=-signal_length + i)))
    return diags


def _test_optimization_template(data_type, signal_length, approximation_rank, seed):
    rng = Generator(PCG64(seed))
    exact_covariance, eigenvectors, eigenvalues = generate_covariance(signal_length, approximation_rank, data_type, rng)
    exact_cov_fourier_basis: Matrix = np.conj(np.fft.fft(exact_covariance, axis=0, norm="ortho").T)
    exact_cov_fourier_basis: Matrix = np.conj(np.fft.fft(exact_cov_fourier_basis, axis=0, norm="ortho").T)
    exact_power_spectrum: Vector = np.real(np.diag(exact_cov_fourier_basis))
    exact_tri_spectrum: ThreeDMatrix = calc_exact_tri_spectrum(exact_cov_fourier_basis, data_type)
    diagonals = find_diagonals(exact_cov_fourier_basis)
    g_mats = np.array([diagonal.reshape(-1, 1).dot(np.conj(diagonal.reshape(-1, 1).T)) for diagonal in diagonals])
    optimization_object: Callable = create_optimization_objective(exact_tri_spectrum, exact_power_spectrum,
                                                                  data_type, use_cp=False)
    return optimization_object(g_mats), exact_tri_spectrum, exact_power_spectrum, g_mats, exact_cov_fourier_basis


class TestOptimization(unittest.TestCase):
    """
    A class which contains tests for the optimization step of Algorithm 1.
    """
    @unittest.skip
    def test_optimization_complex_case(self):
        """
        Test the optimization objective function for the complex case.

        This test creates the optimal input matrices for the objective function
        and then verifies the value of the objective function, given these matrices as input,
        is very close to zero. In addition, this function verifies the theoretical equation between
        the main "diagonal" of the tri-spectrum and two times the square of the power-spectrum squared.
        """
        seed: int = 1995
        data_type = np.complex128
        signal_length: int = 3
        approximation_rank: int = signal_length
        min_value, exact_tri_spectrum, exact_power_spectrum, _, _ = _test_optimization_template(
            data_type, signal_length, approximation_rank, seed)
        tri_diagonal: Vector = np.array([np.real(exact_tri_spectrum[i, i, i]) for i in range(signal_length)],
                                        dtype=np.float64)
        min_expected_value = np.abs(tri_diagonal - 2 * np.power(exact_power_spectrum, 2))
        min_expected_value = np.sum(min_expected_value)
        print(f'Minimal objective function value in the COMPLEX case: {min_value}')
        self.assertTrue(np.allclose(min_value, 0, atol=1e-20, rtol=0), msg=f'Minimal value is NOT optimal={min_value}')
        self.assertEqual(min_expected_value, 0,
                         msg=f'Tri-spectrum and power-spectrum estimation error={min_expected_value}')

    @unittest.skip
    def test_optimization_real_case(self):
        """
        Test the optimization objective function for the real case.

        This test creates the optimal input matrices for the objective function
        and then verifies the value of the objective function, given these matrices as input,
        is very close to zero. In addition, this function verifies the theoretical equation between
        the first term of the tri-spectrum and three times the square of the first entry of the power-spectrum squared.
        """
        seed: int = 1995
        data_type = np.float64
        signal_length: int = 3
        approximation_rank: int = signal_length
        min_value, exact_tri_spectrum, exact_power_spectrum, _, _ = _test_optimization_template(
            data_type, signal_length, approximation_rank, seed)
        tri_diagonal: Vector = np.array([np.real(exact_tri_spectrum[i, i, i]) for i in range(signal_length)],
                                        dtype=np.float64)
        min_expected_value = np.abs(tri_diagonal[0] - 3 * np.power(exact_power_spectrum[0], 2))
        print(f'Minimal objective function value in the REAL case: {min_value}')
        self.assertTrue(np.allclose(min_value, 0, atol=1e-20, rtol=0), msg=f'Minimal value is NOT optimal={min_value}')
        self.assertEqual(min_expected_value, 0, msg=f'Tri-spectrum and power-spectrum estimation error={min_value}')

    def test_estimator_reconstruction(self):
        seed: int = 1995
        data_type = np.float64
        tol = 1e-12
        signal_length: int = 3
        approximation_rank: int = 1
        _, _, exact_power_spectrum, g_mats, exact_cov_fourier = _test_optimization_template(
            data_type, signal_length, approximation_rank, seed)
        g_mats = np.asfortranarray(g_mats)
        exact_power_spectrum = np.ascontiguousarray(exact_power_spectrum)

        diagonals: Matrix = extract_diagonals(g_mats, signal_length)
        estimated_covariance: Matrix = construct_estimator(diagonals, exact_power_spectrum, signal_length, 0)

        # The Hadamard product of this matrix with the exact covariance (in Fourier basis)
        # should be close to the estimator. This matrix should be circulant.
        circulant_coefficient: Matrix = np.divide(estimated_covariance, exact_cov_fourier)
        possible_circulant_mat: Matrix = circulant(circulant_coefficient[:, 0])

        self.assertTrue(np.allclose(possible_circulant_mat, circulant_coefficient, atol=tol, rtol=0),
                        msg=f'The phases matrix is NOT circulant='
                            f'{np.max(np.abs(possible_circulant_mat - circulant_coefficient))}')
        self.assertTrue(np.allclose(np.abs(circulant_coefficient[:, 0]), np.ones(signal_length), atol=tol, rtol=0))


def construct_estimator(diagonals, power_spectrum, signal_length, noise_power) -> Matrix:
    estimator = np.empty((signal_length, signal_length), dtype=diagonals.dtype)

    # Constructing the main diagonal of the estimator.
    for i in range(signal_length):
        estimator[i, i] = power_spectrum[i] - noise_power

    # Constructing the off-diagonal terms.
    for i in range(1, signal_length):
        for j in range(signal_length):
            column_index = (i + j) % signal_length
            estimator[j, column_index] = diagonals[i - 1, j]

    return estimator


def extract_diagonals(matrices_arr, signal_length) -> Matrix:
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
    diagonals = np.empty((signal_length - 1, signal_length), dtype=matrices_arr.dtype, order="F")

    for i in range(signal_length - 1):
        eigenvalue, eigenvector = eigh(
            matrices_arr[i], overwrite_a=True, overwrite_b=True, check_finite=False,
            eigvals=(signal_length - 1, signal_length - 1))
        first_scaled_eigenvalue = np.sqrt(eigenvalue[0])
        for j in range(signal_length):
            diagonals[i, j] = first_scaled_eigenvalue * eigenvector[j, 0]

    return diagonals


if __name__ == '__main__':
    unittest.main()
