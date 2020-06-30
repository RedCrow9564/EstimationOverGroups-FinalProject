# -*- coding: utf-8 -*-
"""
test_tri_spectrum_estimation.py - tests for data creation methods
=================================================================

This module contains the tests for the data creation in all the examples.

"""
import unittest
import numpy as np
from numpy.random import Generator, PCG64
from Infrastructure import pyximportcpp; pyximportcpp.install(setup_args={"include_dirs": np.get_include()},
                                                              reload_support=True, build_in_temp=True, build_dir=r'.')
from polyspectra_estimation import estimate_trispectrum_naive, estimate_trispectrum_v2
from Infrastructure.utils import Vector, Matrix


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
        rng = Generator(PCG64(1995))
        signal_length: int = 10
        observations: int = 8
        signal: Matrix = np.fft.fft(rng.standard_normal((observations, signal_length)))
        trispectrum_naive = estimate_trispectrum_naive(signal)
        trispectrum_improved = estimate_trispectrum_v2(signal)
        # Validate both tri-spectra are equal
        self.assertTrue(np.allclose(trispectrum_naive, trispectrum_improved))


if __name__ == '__main__':
    unittest.main()
