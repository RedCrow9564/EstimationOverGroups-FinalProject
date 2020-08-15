# -*- coding: utf-8 -*-
"""
test_optimization_step.py - tests for the optimization step components
======================================================================

This module contains the tests for the optimization step in Algorithm 1
in the paper.
"""
import unittest
import numpy as np
from scipy.linalg import circulant
from numpy.random import Generator, PCG64
from Infrastructure.utils import Vector, Matrix, ThreeDMatrix
from Infrastructure import pyximportcpp; pyximportcpp.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
from covariance_estimation import create_optimization_objective, perform_optimization
from data_generation import generate_covariance
from vectorized_actions import change_to_fourier_basis, construct_estimator, extract_diagonals
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
    exact_cov_fourier_basis: Matrix = change_to_fourier_basis(exact_covariance)
    exact_power_spectrum: Vector = np.real(np.diag(exact_cov_fourier_basis))
    exact_tri_spectrum: ThreeDMatrix = calc_exact_tri_spectrum(exact_cov_fourier_basis, data_type)
    diagonals = find_diagonals(exact_cov_fourier_basis)
    g_mats = np.array([diagonal.reshape(-1, 1).dot(np.conj(diagonal.reshape(-1, 1).T)) for diagonal in diagonals])
    g_mats = np.vstack((np.outer(exact_power_spectrum, exact_power_spectrum).reshape(1, 3, 3), g_mats))
    optimization_object, _, _ = create_optimization_objective(exact_tri_spectrum, exact_power_spectrum,
                                                              data_type, use_cp=False)
    return optimization_object(g_mats), exact_tri_spectrum, exact_power_spectrum, g_mats, exact_cov_fourier_basis


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
        """
        Test the output of the optimization step

        This test estimates the matrices Gk (assuming no noise) and creates the estimator Cx.
        This test verifies Cx is a circulant matrix, and that its first column is a vector of elements
        in the complex unit circle (i.e complex numbers with magnitude 1).
        """
        seed: int = 1995
        data_type = np.float64
        tol = 1e-12
        signal_length: int = 3
        approximation_rank: int = 1
        _, _, exact_power_spectrum, g_mats, exact_cov_fourier = _test_optimization_template(
            data_type, signal_length, approximation_rank, seed)
        g_mats = np.asfortranarray(g_mats[1:])
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

    def test_optimization_exact_input(self):
        """
        Test the optimization step accuracy.

        This test creates the exact input power-spectrum and tri-spectrum and then performs
        the optimization step of the algorithm. Since the optimizer is not perfect, the error
        is at most 5e-7, instead of the expected perfect-fit score 0. This test is performed for both
        real data and complex data.
        """
        seed: int = 1995
        optimization_error_upper_bound: float = 5e-7
        signal_length: int = 5
        approximation_rank: int = 2
        rng = Generator(PCG64(seed))

        for data_type in [np.complex128, np.float64]:
            exact_cov: Matrix = generate_covariance(signal_length, approximation_rank, data_type, rng)[0]
            exact_cov_fourier_basis: Matrix = change_to_fourier_basis(exact_cov)
            exact_power_spectrum: Vector = np.ascontiguousarray(np.real(np.diag(exact_cov_fourier_basis)))
            exact_tri_spectrum: ThreeDMatrix = calc_exact_tri_spectrum(exact_cov_fourier_basis, data_type)

            g, min_fit_score = perform_optimization(exact_tri_spectrum, exact_power_spectrum, signal_length, data_type)
            min_fit_score = abs(min_fit_score)
            self.assertLess(min_fit_score, optimization_error_upper_bound)
            print(f'Optimization error for exact data of type {data_type} is smaller than '
                  f'{optimization_error_upper_bound}')


if __name__ == '__main__':
    unittest.main()
