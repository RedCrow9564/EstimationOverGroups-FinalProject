# -*- coding: utf-8 -*-
"""
test_diagonal_extraction_and_construction.py - tests for data creation methods
===============================================================================

This module contains the tests for the diagonals' extraction method and the estimator construction method
for the first part of the algorithm.

"""
import unittest
import numpy as np
from Infrastructure import pyximportcpp; pyximportcpp.install(setup_args={"include_dirs": np.get_include()},
                                                              reload_support=True)
from vectorized_actions import extract_diagonals, construct_estimator
from Infrastructure.utils import Vector, Matrix


class TestDiagonalExtractionAndConstruction(unittest.TestCase):
    """
    A class which contains tests for the diagonals extraction method, and the method for constructing
    a matrix, given these diagonals
    """
    def test_diagonals_extraction(self):
        """
        Test diagonals extraction

        This test validates the extract_diagonals take the maximal eigenvector from each matrix in its first dimension
        and scales it w.r.t square-root of the maximal eigenvalue.

        """
        mat_num: int = 4
        max_eigenvalue: float = 9.0
        sqrt_max_eigenvalue: float = np.sqrt(max_eigenvalue)
        mat: Matrix = np.asfortranarray(np.tile(np.diag([3.0, max_eigenvalue, -4.0, -2.0]), (mat_num - 1, 1, 1)))
        diagonals: Matrix = extract_diagonals(mat, mat_num)
        expected_output: Matrix = np.tile(np.array([0, sqrt_max_eigenvalue, 0, 0]), (3, 1))
        self.assertTrue(np.allclose(diagonals, expected_output))

    def test_estimator_construction(self):
        """
        Test methods equivalence.

        This test validates that both the naive method and the improved method return identical
        output (up to numerical inaccuracies) for some random signal.

        """
        mat_num: int = 4
        max_eigenvalue: float = 9.0
        sqrt_max_eigenvalue: float = np.sqrt(max_eigenvalue)
        mat: Matrix = np.asfortranarray(np.tile(np.diag([3.0, max_eigenvalue, -4.0, -2.0]), (mat_num - 1, 1, 1)))
        diagonals: Matrix = extract_diagonals(mat, mat_num)
        power_spectrum: Vector = np.array(4 * [1.0])
        estimator: Matrix = construct_estimator(diagonals, power_spectrum, 4, 0)
        expected_output: Matrix = np.eye(4)
        expected_output[1] = np.array([sqrt_max_eigenvalue, 1, sqrt_max_eigenvalue, sqrt_max_eigenvalue])
        self.assertTrue(np.allclose(estimator, expected_output))


if __name__ == '__main__':
    unittest.main()
