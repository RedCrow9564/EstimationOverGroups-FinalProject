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
from polyspectra_estimation import estimate_tri_spectrum_naive, estimate_tri_spectrum_v2
from Infrastructure.utils import Matrix, ThreeDMatrix


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
        signal_length: int = 10
        observations: int = 8
        signal: Matrix = np.fft.fft(rng.standard_normal((observations, signal_length)))
        tri_spectrum_naive: ThreeDMatrix = estimate_tri_spectrum_naive(signal)
        tri_spectrum_improved: ThreeDMatrix = estimate_tri_spectrum_v2(signal)
        # Validate both tri-spectra are equal
        self.assertTrue(np.allclose(np.abs(tri_spectrum_naive), np.abs(tri_spectrum_improved)),
                        msg=f'{np.max(np.abs(tri_spectrum_improved - tri_spectrum_naive))}')
        self.assertTrue(np.allclose(np.angle(tri_spectrum_naive), np.angle(tri_spectrum_improved)),
                        msg=f'{np.max(np.angle(tri_spectrum_improved) - np.angle(tri_spectrum_naive))}')


if __name__ == '__main__':
    unittest.main()
