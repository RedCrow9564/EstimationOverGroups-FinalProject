import numpy as np
from numpy.fft import ifft
from scipy.linalg import eigh, block_diag, circulant
from scipy.sparse.linalg import eigsh
from Infrastructure.utils import ex, Union, List, Scalar, Vector, Matrix
from polyspectra_estimation import estimate_power_spectrum, estimate_trispectrum_v2
from vectorized_actions import extract_diagonals, construct_estimator, vectorized_kron


def low_rank_multi_reference_factor_analysis(
        observations_fourier: Matrix, signal_length: int, approximation_rank: Union[int, None],
        noise_power: float, data_type, exact_covariance) -> Matrix:
    estimator: Matrix = estimate_covariance_up_to_phases(observations_fourier, signal_length, noise_power, data_type, exact_covariance)
    estimator = phase_retrieval(estimator, signal_length, approximation_rank)
    return estimator


def estimate_covariance_up_to_phases(observations_fourier: Matrix, signal_length: int, noise_power: float,
                                     data_type, exact_covariance) -> Matrix:
    power_spectrum: Vector = estimate_power_spectrum(observations_fourier)
    tri_spectrum = estimate_trispectrum_v2(observations_fourier)

    # TODO: Implement the optimization step for both real and complex data.
    g = np.asfortranarray(np.tile(np.eye(signal_length, dtype=data_type), (signal_length - 1, 1, 1)))

    diagonals: Matrix = extract_diagonals(g, signal_length)
    estimated_covariance: Matrix = construct_estimator(diagonals, power_spectrum, signal_length, noise_power)
    return exact_covariance  # TODO: Replace with estimated_covariance


def phase_retrieval(covariance_estimator: Matrix, signal_length: int, approximation_rank: Union[int, None]) -> Matrix:
    """

    """

    # Building the coefficients matrix A.
    matrix_a = coefficient_matrix_construction(covariance_estimator, signal_length, approximation_rank)
    # Finding the singular vector which corresponds to the smallest singular-value of A.
    b = np.dot(np.conj(matrix_a).T, matrix_a)
    print(f'{b.shape}')
    val, v = eigh(b, overwrite_a=False, overwrite_b=False, check_finite=False, eigvals=(0, 1))

    # Finding the matching angles
    v = v[:, 1][:signal_length]
    arguments_vector = np.angle(v)
    angles = np.cumsum(arguments_vector[1:])
    angles = -angles + (angles[signal_length - 2] + arguments_vector[0]) / signal_length * \
        np.arange(1, signal_length, 1)
    print(angles)
    phases: Vector = np.exp(-1j * angles)
    phases = np.insert(phases, 0, 1)
    print(phases)

    # Multiplying by the negative phases to find the Fourier-basis covariance
    covariance_estimator = np.multiply(covariance_estimator, circulant(phases))

    # Converting the estimator back to the standard basis
    covariance_estimator = np.conj(ifft(covariance_estimator, axis=0, norm="ortho").T)
    covariance_estimator = np.conj(ifft(covariance_estimator, axis=0, norm="ortho").T)
    return covariance_estimator


def coefficient_matrix_construction(covariance_estimator: Matrix, signal_length: int,
                                    approximation_rank: Union[int, None]) -> Matrix:
    hi_i = np.power(np.abs(covariance_estimator), 2)
    rotated_mat = covariance_estimator.copy()
    rank_sqr = approximation_rank ** 2 if approximation_rank is not None else signal_length - 1
    least_eigenvalue_index = signal_length - rank_sqr
    all_eigenvectors = np.empty((signal_length, signal_length, rank_sqr), order="F")
    all_m_mats = np.empty((signal_length ** 3, signal_length))

    first_indices = np.arange(0, signal_length ** 2, signal_length + 1)
    for i in range(signal_length):
        all_eigenvectors[i] = eigh(hi_i, overwrite_b=False, check_finite=False,
                                   eigvals=(least_eigenvalue_index, signal_length - 1))[1]
        rotated_mat = np.roll(rotated_mat, shift=1, axis=1)
        hi_i = np.multiply(covariance_estimator, np.conj(rotated_mat))
        transition_index: int = 0
        last_index: int = signal_length ** 2

        for m in range(signal_length):
            if m == 0:
                sampled_hi_indices = first_indices.copy()
            else:
                transition_index += signal_length
                sampled_hi_indices = np.arange(signal_length - m, transition_index, signal_length + 1)
                last_index -= 1
                sampled_hi_indices = np.hstack((sampled_hi_indices, np.arange(transition_index,
                                                                              last_index, signal_length + 1)))
            all_m_mats[sampled_hi_indices + i * signal_length ** 2, m] = hi_i.flatten('F')[sampled_hi_indices]

        rotated_mat = np.roll(rotated_mat, shift=1, axis=0)
        hi_i = np.multiply(covariance_estimator, np.conj(np.roll(rotated_mat, shift=1, axis=0)))

    matrix_a = np.hstack((all_m_mats, -block_diag(*vectorized_kron(all_eigenvectors))))
    return matrix_a
