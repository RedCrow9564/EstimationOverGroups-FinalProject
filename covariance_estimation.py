import numpy as np
from numpy.fft import ifft
from scipy.linalg import eigh, block_diag, circulant
from scipy.sparse.linalg import eigsh
import cvxpy as cp
from Infrastructure.utils import Union, List, Scalar, Vector, Matrix, ThreeDMatrix, Callable
from polyspectra_estimation import estimate_power_spectrum, estimate_trispectrum_v2
from vectorized_actions import extract_diagonals, construct_estimator, vectorized_kron


def low_rank_multi_reference_factor_analysis(
        observations_fourier: Matrix, signal_length: int, approximation_rank: Union[int, None],
        noise_power: float, data_type, exact_covariance) -> Matrix:
    # TODO: Remove the exact_covariance argument for real experiments.
    estimator: Matrix = estimate_covariance_up_to_phases(observations_fourier, signal_length, noise_power, data_type,
                                                         exact_covariance)
    estimator = phase_retrieval(estimator, signal_length, approximation_rank)
    return estimator


def estimate_covariance_up_to_phases(observations_fourier: Matrix, signal_length: int, noise_power: float,
                                     data_type, exact_covariance) -> Matrix:
    power_spectrum: Vector = estimate_power_spectrum(observations_fourier)
    tri_spectrum: ThreeDMatrix = estimate_trispectrum_v2(observations_fourier)

    # Optimization step
    # g = perform_optimization(tri_spectrum, power_spectrum, signal_length, data_type)
    # TODO: Implement the optimization step for both real and complex data.
    g = np.asfortranarray(np.tile(np.eye(signal_length, dtype=data_type), (signal_length - 1, 1, 1)))

    diagonals: Matrix = extract_diagonals(g, signal_length)
    estimated_covariance: Matrix = construct_estimator(diagonals, power_spectrum, signal_length, noise_power)
    return exact_covariance  # TODO: Replace with estimated_covariance


def create_optimization_objective(tri_spectrum, power_spectrum, data_type) -> Callable:
    g_zero = np.outer(power_spectrum, power_spectrum)
    signal_length = len(power_spectrum)

    def objective(matrices_list) -> float:
        fit_score = 0.0
        for k1 in range(signal_length):
            for k2 in range(signal_length):
                other_index = (k2 - k1) % signal_length
                for m in range(signal_length):
                    k1_plus_m = (k1 + m) % signal_length
                    current_term = tri_spectrum[k1, k1_plus_m, (k2 + m) % signal_length]
                    if other_index == 0:
                        current_term -= g_zero[k1, k1_plus_m]
                    else:
                        current_term -= matrices_list[other_index - 1][k1, k1_plus_m]
                    if m == 0:
                        current_term -= g_zero[k1, k2]
                    else:
                        current_term -= matrices_list[m - 1][k1, k2]

                    if data_type in [np.float32, np.float64]:
                        next_index: int = (k1 + k2 + m) % (signal_length - 1)
                        if next_index == 0:
                            current_term -= g_zero[- k2 % signal_length, (-k2 - m) % signal_length]
                        else:
                            current_term -= matrices_list[next_index - 1][
                                - k2 % signal_length, (-k2 - m) % signal_length]

                    fit_score += cp.abs(current_term) ** 2
        return fit_score
    return objective


def perform_optimization(tri_spectrum: ThreeDMatrix, power_spectrum: Matrix, signal_length: int,
                         data_type) -> ThreeDMatrix:
    optimization_objective: Callable = create_optimization_objective(tri_spectrum, power_spectrum, data_type)
    symbolic_matrices = [cp.Variable((signal_length, signal_length), PSD=True) for _ in range(signal_length - 1)]
    problem = cp.Problem(cp.Minimize(optimization_objective(symbolic_matrices)), [])
    min_fit_score = problem.solve(verbose=True)
    print(f'Min score: {min_fit_score}')
    print([mat.value for mat in symbolic_matrices])
    return np.array([mat.value for mat in symbolic_matrices], order='F')


def phase_retrieval(covariance_estimator: Matrix, signal_length: int, approximation_rank: Union[int, None]) -> Matrix:
    """

    """

    # Building the coefficients matrix A.
    matrix_a: Matrix = coefficient_matrix_construction(covariance_estimator, signal_length, approximation_rank)
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
    hi_i: Matrix = np.power(np.abs(covariance_estimator), 2)
    rotated_mat: Matrix = covariance_estimator.copy()
    rank_sqr: int = approximation_rank ** 2 if approximation_rank is not None else signal_length - 1
    least_eigenvalue_index: int = signal_length - rank_sqr
    all_eigenvectors: Matrix = np.empty((signal_length, signal_length, rank_sqr), order="F",
                                        dtype=covariance_estimator.dtype)
    all_m_mats: Matrix = np.empty((signal_length ** 3, signal_length), dtype=covariance_estimator.dtype)

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
