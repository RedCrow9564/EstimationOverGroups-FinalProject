# -*- coding: utf-8 -*-
"""
test_tri_spectrum_estimation.py - tests for spectra estimation methods
======================================================================

This module contains the tests for the power-spectrum and tri-spectrum estimation methods.
"""
import unittest
import numpy as np
from numpy.random import Generator, PCG64
from itertools import product
from Infrastructure import pyximportcpp; pyximportcpp.install(setup_args={"include_dirs": np.get_include()},
                                                              reload_support=True)
from polyspectra_estimation import estimate_tri_spectrum_naive, estimate_tri_spectrum_v2, estimate_power_spectrum
from Infrastructure.utils import Vector, Matrix, ThreeDMatrix
from vectorized_actions import change_to_fourier_basis
from data_generation import generate_covariance, generate_observations


class TestTriSpectrumAlgorithms(unittest.TestCase):
    """
    A class which contains tests for the tri-spectrum estimation algorithms.
    """
    def test_equivalence_to_naive_method(self):
        """
        Test methods equivalence.

        This test validates that both the naive method and the improved method return identical
        output (up to numerical inaccuracies) for some random signal.

        """
        rng = Generator(PCG64(596))
        signal_length: int = rng.integers(low=10, high=40)
        observations_num: int = rng.integers(low=5, high=20)
        approximation_rank: int = rng.integers(low=5, high=20)
        data_type = np.complex128
        covariance, eigenvectors, eigenvalues = generate_covariance(signal_length, approximation_rank, data_type, rng)
        observations = generate_observations(eigenvectors, eigenvalues, approximation_rank, observations_num,
                                             data_type, rng)
        observations_fourier: Matrix = np.fft.fft(observations, axis=1, norm="ortho")
        tri_spectrum_naive: ThreeDMatrix = estimate_tri_spectrum_naive(observations_fourier)
        tri_spectrum_improved: ThreeDMatrix = estimate_tri_spectrum_v2(observations_fourier)
        # Validate both tri-spectra are equal
        self.assertTrue(np.allclose(np.abs(tri_spectrum_naive), np.abs(tri_spectrum_improved)),
                        msg=f'{np.max(np.abs(tri_spectrum_improved - tri_spectrum_naive))}')
        self.assertTrue(np.allclose(np.angle(tri_spectrum_naive), np.angle(tri_spectrum_improved)),
                        msg=f'{np.max(np.angle(tri_spectrum_improved) - np.angle(tri_spectrum_naive))}')

    def test_power_spectrum_and_tri_spectrum_consistency(self):
        """
        Test the consistency of the power-spectrum estimation.

        This test validates that the estimation over a large number of observations (without noise)
        is "very close" to the exact power-spectrum

        """
        rng = Generator(PCG64(596))
        signal_length: int = rng.integers(low=10, high=20)
        observations_num: int = rng.integers(low=10000, high=50000)
        approximation_rank: int = rng.integers(low=2, high=signal_length)
        data_type = np.complex128
        tol = 1e-3
        exact_covariance, eigenvectors, eigenvalues = generate_covariance(signal_length, approximation_rank, data_type,
                                                                          rng)
        observations = generate_observations(eigenvectors, eigenvalues, approximation_rank, observations_num,
                                             data_type, rng)
        observations_fourier: Matrix = np.fft.fft(observations, norm="ortho", axis=1)
        exact_cov_fourier_basis: Matrix = change_to_fourier_basis(exact_covariance)
        exact_diagonal: Vector = np.real(np.diag(exact_cov_fourier_basis))
        power_spectrum_estimation: Vector = estimate_power_spectrum(observations_fourier)
        exact_tri_spectrum: ThreeDMatrix = calc_exact_tri_spectrum(exact_cov_fourier_basis, data_type)
        estimate_tri_spectrum: ThreeDMatrix = estimate_tri_spectrum_v2(observations_fourier)

        print(f'Observations number: {observations_num}')
        print(f'Complex Tri-spectrum estimation error: {np.max(np.abs(estimate_tri_spectrum - exact_tri_spectrum))}')

        # Validate both tri-spectrum and power-spectrum estimations are consistent.
        self.assertTrue(np.allclose(exact_diagonal, power_spectrum_estimation, atol=tol, rtol=0),
                        msg=f'Power-spectrum estimation is inconsistent!, error=' +
                            f'{np.max(np.abs(power_spectrum_estimation - exact_diagonal))}')
        self.assertTrue(np.allclose(estimate_tri_spectrum, exact_tri_spectrum, atol=tol, rtol=0),
                        msg=f'Tri-spectrum estimation is inconsistent!, error=' +
                            f'{np.max(np.abs(estimate_tri_spectrum - exact_tri_spectrum))}')

    def test_real_data_tri_spectrum_consistency(self):
        """
        Test the consistency of the tri-spectrum estimation in the case of REAL data (since the exact
        tri-spectrum becomes more complicated in this case).

        This test validates that the estimation over a large number of observations (without noise)
        is "very close" to the exact tri-spectrum

        """
        rng = Generator(PCG64(596))
        signal_length: int = rng.integers(low=10, high=20)
        observations_num: int = rng.integers(low=10000, high=40000)
        approximation_rank: int = rng.integers(low=2, high=signal_length)
        data_type = np.float64
        tol = 1e-3
        exact_covariance, eigenvectors, eigenvalues = generate_covariance(signal_length, approximation_rank, data_type,
                                                                          rng)
        observations = generate_observations(eigenvectors, eigenvalues, approximation_rank, observations_num,
                                             data_type, rng)
        observations_fourier: Matrix = np.fft.fft(observations, norm="ortho")
        exact_cov_fourier_basis: Matrix = change_to_fourier_basis(exact_covariance)
        exact_tri_spectrum: ThreeDMatrix = calc_exact_tri_spectrum(exact_cov_fourier_basis, data_type)
        estimate_tri_spectrum: ThreeDMatrix = estimate_tri_spectrum_v2(observations_fourier)
        
        print(f'Observations number: {observations_num}')
        print(f'Real Tri-spectrum estimation error: {np.max(np.abs(estimate_tri_spectrum - exact_tri_spectrum))}')

        # Validate the tri-spectrum estimation in the real case is consistent.
        self.assertTrue(np.allclose(estimate_tri_spectrum, exact_tri_spectrum, atol=tol, rtol=0),
                        msg=f'Tri-spectrum estimation is inconsistent!, error=' +
                            f'{np.max(np.abs(estimate_tri_spectrum - exact_tri_spectrum))}')


def calc_exact_tri_spectrum(exact_covariance: Matrix, data_type) -> ThreeDMatrix:
    signal_length: int = exact_covariance.shape[0]
    tri_spectrum = np.empty((signal_length, signal_length, signal_length), dtype=exact_covariance.dtype)

    for i, j, k in product(range(signal_length), repeat=3):
        fourth_index = (k - j + i) % signal_length
        tri_spectrum[i, j, k] = exact_covariance[i, j] * np.conj(exact_covariance[fourth_index, k])
        tri_spectrum[i, j, k] += exact_covariance[i, fourth_index] * np.conj(exact_covariance[j, k])

        if data_type in [np.float32, np.float64]:
            tri_spectrum[i, j, k] += exact_covariance[i, (-k % signal_length)] * np.conj(
                exact_covariance[j, (-fourth_index % signal_length)])

    return tri_spectrum


if __name__ == '__main__':
    unittest.main()
