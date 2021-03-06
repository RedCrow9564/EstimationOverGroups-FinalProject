# -*- coding: utf-8 -*-
"""
test_phase_retrieval_actions.py - tests for the stages in the phase retrieval algorithm
=======================================================================================

This module contains the tests for the different stages of the phase-retrieval algorithm:
constructing the coefficients matrix and the Fourier basis transition of the fixed covariance estimator.

"""
import unittest
import numpy as np
from numpy.random import Generator, PCG64
from scipy.linalg import dft, eigh, block_diag, circulant
from Infrastructure.utils import Union, Vector, Matrix
from data_generation import generate_covariance
from main import calc_estimation_error
from covariance_estimation import coefficient_matrix_construction, phase_retrieval
from vectorized_actions import vectorized_kron, change_to_fourier_basis, change_from_fourier_basis


class TestPhaseRetrievalActions(unittest.TestCase):
    """
    A class which tests components of the phase-retrieval algorithm
    """
    def test_fourier_matrices_product(self):
        """
        Test Fourier basis transition

        This test validates that the transition of a matrix from Fourier basis to the standard basis is equivalent to
        multiplying my the DFT matrix from right and its inverse from the left.

        """
        rng = Generator(PCG64(1995))
        n: int = rng.integers(low=2, high=100)
        mat: Matrix = rng.standard_normal((n, n))
        tested_output = change_from_fourier_basis(mat)
        dft_mat: Matrix = dft(n, scale="sqrtn")
        expected_output: Matrix = np.conj(dft_mat.T).dot(mat).dot(dft_mat)
        self.assertTrue(np.allclose(tested_output, expected_output))

        mat: Matrix = rng.standard_normal((n, n))
        mat_fourier: Matrix = change_to_fourier_basis(mat)
        expected_output: Matrix = dft_mat.dot(mat).dot(np.conj(dft_mat.T))
        self.assertTrue(np.allclose(mat_fourier, expected_output))

    def test_coefficient_matrix_construction(self):
        """
        Test coefficient matrix properties

        This test validates that the coefficients matrix in the phase retrieval algorithm follows the
        theoretical properties.

        """
        rng = Generator(PCG64(1995))
        n: int = rng.integers(low=9, high=20)
        r: int = rng.integers(low=2, high=np.floor(np.sqrt(n)))
        mat: Matrix = generate_covariance(n, r, np.complex128, rng)[0]
        mat = change_to_fourier_basis(mat)
        
        coeff_mat: Matrix = coefficient_matrix_construction(mat, signal_length=n, approximation_rank=r)
        self.assertEqual(coeff_mat.shape[0], n ** 3)
        self.assertEqual(coeff_mat.shape[1], (1 + r ** 4) * n)

        other_mat: Matrix = matrix_construction_alternative(mat, signal_length=n, approximation_rank=r)
        self.assertTrue(np.allclose(coeff_mat, other_mat), msg=f'{np.max(np.abs(other_mat - coeff_mat))}')

    def test_phase_retrieval_on_exact_data(self):
        """
        Test phase-retrieval performance on exact data.

        This test performs the phase-retrieval on the exact covariance, up to some random integer shift.
        The output estimated covariance should be equal to the exact covariance, up to some shift of both its axes,
        i.e the estimation error should be very close to zero (about 10^-20).
        This test is performed for both real and complex data.
        """
        rng = Generator(PCG64(1995))
        signal_length: int = rng.integers(low=9, high=20)
        approximation_rank: int = rng.integers(low=2, high=np.floor(np.sqrt(signal_length)))
        tol: float = 1e-20

        for data_type in [np.complex128, np.float64]:
            exact_cov: Matrix = generate_covariance(signal_length, approximation_rank, data_type, rng)[0]
            random_shift: int = rng.integers(low=0, high=signal_length)
            rotated_cov: Matrix = np.roll(exact_cov, shift=[random_shift, random_shift], axis=[0, 1])
            exact_cov_fourier_basis: Matrix = change_to_fourier_basis(rotated_cov)

            random_phases: Vector = 1j * rng.standard_normal(signal_length - 1)
            input_cov = np.multiply(exact_cov_fourier_basis, circulant([1] + np.exp(random_phases).tolist()))     
            estimated_cov: Matrix = phase_retrieval(input_cov, signal_length, approximation_rank)
            estimation_error: float = calc_estimation_error(exact_cov, estimated_cov)

            self.assertTrue(np.allclose(estimation_error, 0, atol=tol, rtol=0))


def matrix_construction_alternative(covariance_estimator: Matrix, signal_length: int,
                                    approximation_rank: Union[int, None]) -> Matrix:
    hi_i: Matrix = np.power(np.abs(covariance_estimator), 2)
    rotated_mat: Matrix = covariance_estimator.copy()
    rank_sqr: int = approximation_rank ** 2 if approximation_rank is not None else signal_length - 1
    least_eigenvalue_index: int = signal_length - rank_sqr
    last_index: int = signal_length ** 2
    all_eigenvectors: Matrix = np.empty((signal_length, signal_length, rank_sqr), order="F",
                                        dtype=covariance_estimator.dtype)
    all_m_mats: Matrix = np.empty((signal_length ** 3, signal_length), dtype=covariance_estimator.dtype)

    for i in range(signal_length):
        all_eigenvectors[i] = eigh(hi_i, overwrite_b=False, check_finite=False,
                                   eigvals=(least_eigenvalue_index, signal_length - 1))[1]
        rotated_mat = np.roll(rotated_mat, shift=1, axis=1)
        hi_i = np.multiply(covariance_estimator, np.conj(rotated_mat))
        hi_i_plus_one_flat = hi_i.flatten('F')

        for m in range(signal_length):
            em = np.zeros(signal_length)
            em[m] = 1
            circ = circulant(em).flatten('F')
            all_m_mats[i * last_index : (i + 1) * last_index, m] = np.multiply(hi_i_plus_one_flat, circ)
        rotated_mat = np.roll(rotated_mat, shift=1, axis=0)
        hi_i = np.multiply(covariance_estimator, np.conj(rotated_mat))

    matrix_a = np.hstack((all_m_mats, -block_diag(*vectorized_kron(all_eigenvectors))))
    return matrix_a


if __name__ == '__main__':
    unittest.main()
