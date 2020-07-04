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
from numpy.fft import ifft
from scipy.linalg import dft, eigh, block_diag, circulant
from Infrastructure.utils import Matrix, Union
from covariance_estimation import coefficient_matrix_construction
from vectorized_actions import vectorized_kron


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
        tested_output = np.conj(ifft(mat, axis=0, norm="ortho").T)
        tested_output = np.conj(ifft(tested_output, axis=0, norm="ortho").T)
        dft_mat: Matrix = dft(n, scale="sqrtn")
        expected_output: Matrix = np.conj(dft_mat.T).dot(mat).dot(dft_mat)
        self.assertTrue(np.allclose(tested_output, expected_output))

    def test_coefficient_matrix_construction(self):
        """
        Test coefficient matrix properties

        This test validates that the coefficients matrix in the phase retrieval algorithm follows the
        theoretical properties.

        """
        rng = Generator(PCG64(1995))
        n: int = rng.integers(low=9, high=20)
        r: int = rng.integers(low=2, high=np.floor(np.sqrt(n)))
        mat: Matrix = rng.standard_normal((n, n))
        mat = np.dot(mat, mat.T)
        coeff_mat: Matrix = coefficient_matrix_construction(mat, signal_length=n, approximation_rank=r)
        self.assertEqual(coeff_mat.shape[0], n ** 3)
        self.assertEqual(coeff_mat.shape[1], (1 + r ** 4) * n)

        other_mat: Matrix = matrix_construction_alternative(mat, signal_length=n, approximation_rank=r)
        self.assertTrue(np.allclose(coeff_mat, other_mat), msg=f'{np.max(np.abs(other_mat - coeff_mat))}')


def matrix_construction_alternative(covariance_estimator: Matrix, signal_length: int,
                                    approximation_rank: Union[int, None]) -> Matrix:
    hi_i: Matrix = np.power(np.abs(covariance_estimator), 2)
    rotated_mat: Matrix = covariance_estimator.copy()
    rank_sqr: int = approximation_rank ** 2 if approximation_rank is not None else signal_length - 1
    least_eigenvalue_index: int = signal_length - rank_sqr
    last_index: int = signal_length ** 2
    all_eigenvectors: Matrix = np.empty((signal_length, signal_length, rank_sqr), order="F",
                                        dtype=covariance_estimator.dtype)
    all_m_mats: Matrix = np.zeros((signal_length ** 3, signal_length), dtype=covariance_estimator.dtype)

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
