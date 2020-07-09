# -*- coding: utf-8 -*-
"""
test_tri_spectrum_estimation.py - tests for spectra estimation methods
======================================================================

This module contains the tests for the power-spectrum and tri-spectrum estimation methods.
"""
import unittest
import numpy as np
from numpy.random import Generator, PCG64
from Infrastructure import pyximportcpp; pyximportcpp.install(setup_args={"include_dirs": np.get_include()},
                                                              reload_support=True)
from polyspectra_estimation import estimate_tri_spectrum_naive, estimate_tri_spectrum_v2, estimate_power_spectrum
from Infrastructure.utils import Vector, Matrix, ThreeDMatrix
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

    def test_power_spectrum_consistency(self):
        """
        Test the consistency of the power-spectrum estimation.

        This test validates that the estimation over a large number of observations (without noise)
        is "very close" to the exact power-spectrum

        """
        rng = Generator(PCG64(596))
        signal_length: int = rng.integers(low=10, high=40)
        observations_num: int = rng.integers(low=100000, high=500000)
        approximation_rank: int = rng.integers(low=5, high=20)
        data_type = np.complex128
        tol = 1e-3
        exact_covariance, eigenvectors, eigenvalues = generate_covariance(signal_length, approximation_rank, data_type,
                                                                          rng)
        observations = generate_observations(eigenvectors, eigenvalues, approximation_rank, observations_num,
                                             data_type, rng)
        observations_fourier: Matrix = np.fft.fft(observations, norm="ortho")
        exact_cov_fourier_basis: Matrix = np.conj(np.fft.fft(exact_covariance, axis=0, norm="ortho").T)
        exact_cov_fourier_basis: Matrix = np.conj(np.fft.fft(exact_cov_fourier_basis, axis=0, norm="ortho").T)
        exact_diagonal: Vector = np.real(np.diag(exact_cov_fourier_basis))
        power_spectrum_estimation: Vector = estimate_power_spectrum(observations_fourier)

        # Validate both tri-spectra are equal
        self.assertTrue(np.allclose(exact_diagonal, power_spectrum_estimation, atol=tol, rtol=0),
                        msg=f'Power-spectrum estimation is inconsistent!')


if __name__ == '__main__':
    unittest.main()
