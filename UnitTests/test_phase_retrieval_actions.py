import unittest
import numpy as np
from numpy.random import Generator, PCG64
from numpy.fft import ifft
from scipy.linalg import dft
from Infrastructure.utils import Matrix
from covariance_estimation import coefficient_matrix_construction


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
        n: int = rng.integers(low=5, high=40)
        r: int = rng.integers(low=2, high=np.floor(np.sqrt(n)))
        mat: Matrix = rng.standard_normal((n, n))
        coeff_mat: Matrix = coefficient_matrix_construction(mat, signal_length=n, approximation_rank=r)
        self.assertEqual(coeff_mat.shape[0], n ** 3)
        self.assertEqual(coeff_mat.shape[1], (1 + r ** 4) * n)


if __name__ == '__main__':
    unittest.main()
