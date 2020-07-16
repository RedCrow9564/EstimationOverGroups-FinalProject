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
from numpy.random import Generator, PCG64
from Infrastructure.utils import Vector, Matrix, ThreeDMatrix
from data_generation import generate_covariance
from UnitTests.test_tri_spectrum_estimation import calc_exact_tri_spectrum


def _create_optimization_objective(tri_spectrum, power_spectrum, data_type) -> Callable:
    g_zero = np.outer(power_spectrum, power_spectrum)
    signal_length = len(power_spectrum)

    def objective(matrices_list):
        fit_score = 0.0
        current_matrices_list = np.concatenate(([g_zero], matrices_list), axis=0)
        for k1 in range(signal_length):
            for k2 in range(signal_length):
                other_index = (k2 - k1) % signal_length
                k1_plus_m = k1
                k2_plus_m = k2
                for m in range(signal_length):
                    current_term = tri_spectrum[k1, k1_plus_m, k2_plus_m]
                    current_term -= current_matrices_list[other_index][k1, k1_plus_m]
                    current_term -= current_matrices_list[m][k1, k2]

                    if data_type in [np.float32, np.float64]:
                        next_index: int = (k1 + k2 + m) % signal_length
                        current_term -= current_matrices_list[next_index][(-k2) % signal_length,
                                                                          (-k2 - m) % signal_length]

                    k1_plus_m = (k1_plus_m + 1) % signal_length
                    k2_plus_m = (k2_plus_m + 1) % signal_length
                    fit_score += abs(current_term) ** 2
        return fit_score
    return objective


def _find_diagonals(exact_covariance: Matrix) -> Matrix:
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
    diagonals = _find_diagonals(exact_cov_fourier_basis)
    g_mats = np.array([diagonal.reshape(-1, 1).dot(np.conj(diagonal.reshape(-1, 1).T)) for diagonal in diagonals])
    optimization_object: Callable = _create_optimization_objective(exact_tri_spectrum, exact_power_spectrum, data_type)
    return optimization_object(g_mats), exact_tri_spectrum, exact_power_spectrum


class TestOptimization(unittest.TestCase):
    """
    A class which contains tests for the optimization step of Algorithm 1.
    """
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
        min_value, exact_tri_spectrum, exact_power_spectrum = _test_optimization_template(
            data_type, signal_length, approximation_rank, seed)
        tri_diagonal: Vector = np.array([np.real(exact_tri_spectrum[i, i, i]) for i in range(signal_length)],
                                        dtype=np.float64)
        min_expected_value = np.abs(tri_diagonal - 2 * np.power(exact_power_spectrum, 2))
        min_expected_value = np.sum(min_expected_value)
        print(f'Minimal objective function value in the COMPLEX case: {min_value}')
        self.assertTrue(np.allclose(min_value, 0, atol=1e-20, rtol=0), msg=f'Minimal value is NOT optimal={min_value}')
        self.assertEqual(min_expected_value, 0,
                         msg=f'Tri-spectrum and power-spectrum estimation error={min_expected_value}')

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
        min_value, exact_tri_spectrum, exact_power_spectrum = _test_optimization_template(data_type, signal_length,
                                                                                          approximation_rank, seed)
        tri_diagonal: Vector = np.array([np.real(exact_tri_spectrum[i, i, i]) for i in range(signal_length)],
                                        dtype=np.float64)
        min_expected_value = np.abs(tri_diagonal[0] - 3 * np.power(exact_power_spectrum[0], 2))
        print(f'Minimal objective function value in the REAL case: {min_value}')
        self.assertTrue(np.allclose(min_value, 0, atol=1e-20, rtol=0), msg=f'Minimal value is NOT optimal={min_value}')
        self.assertEqual(min_expected_value, 0, msg=f'Tri-spectrum and power-spectrum estimation error={min_value}')


if __name__ == '__main__':
    unittest.main()
