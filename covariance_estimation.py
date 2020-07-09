import numpy as np
from numpy.fft import ifft, fft
from scipy.linalg import eigh, block_diag, circulant
import cvxpy as cp
import warnings
from Infrastructure.utils import Union, Vector, Matrix, ThreeDMatrix, Callable
from polyspectra_estimation import estimate_power_spectrum, estimate_tri_spectrum_v2
from vectorized_actions import extract_diagonals, construct_estimator, vectorized_kron


def low_rank_multi_reference_factor_analysis(
        observations_fourier: Matrix, signal_length: int, approximation_rank: Union[int, None],
        noise_power: float, data_type, exact_covariance) -> Matrix:
    """
    The entire algorithm fot the covariance estimation. It consists of two stages, see Algorithm 1 and Algorithm 2 in
    the paper.

    Args:
        observations_fourier(Matrix): The fourier coefficients of all observations.
        signal_length(int): The length of the signal.
        approximation_rank(int): The rank of the approximated covariance matrix.
        noise_power(float): An estimator for the noise power in the sampled observations.
        data_type: Either np.float64 for real-data or np.complex128 for complex data.
        exact_covariance(Matrix): The covariance which this algorithm estimated. It is used ONLY for
            testing, debugging and verifying the technical conditions which are required for this algorithm.

    Returns:
        The estimated covariance matrix.

    """
    # TODO: Remove the exact_covariance argument for real experiments.
    estimator: Matrix = estimate_covariance_up_to_phases(observations_fourier, signal_length, noise_power, data_type,
                                                         exact_covariance)
    estimator = phase_retrieval(estimator, signal_length, approximation_rank)
    return estimator


def estimate_covariance_up_to_phases(observations_fourier: Matrix, signal_length: int, noise_power: float,
                                     data_type, exact_covariance) -> Matrix:
    """
    The first stage algorithm fot the covariance estimation, see Algorithm 1 in the paper.

    Args:
        observations_fourier(Matrix): The fourier coefficients of all observations.
        signal_length(int): The length of the signal.
        noise_power(float): An estimator for the noise power in the sampled observations.
        data_type: Either np.float64 for real-data or np.complex128 for complex data.
        exact_covariance(Matrix): The covariance which this algorithm estimated. It is used ONLY for
            testing, debugging and verifying the technical conditions which are required for this algorithm.

    Returns:
        The estimated covariance matrix, up to multiplication by a circulant matrix of complex phases.

    """
    power_spectrum: Vector = estimate_power_spectrum(observations_fourier)
    tri_spectrum: ThreeDMatrix = estimate_tri_spectrum_v2(observations_fourier)

    # TODO: Remove the exact_covariance argument
    exact_cov_fourier_basis: Matrix = np.conj(fft(exact_covariance, axis=0, norm="ortho").T)
    exact_cov_fourier_basis: Matrix = np.conj(fft(exact_cov_fourier_basis, axis=0, norm="ortho").T)
    if np.any(np.abs(exact_covariance)) < 1e-15:
        warnings.warn("The covariance matrix in Fourier basis has some 0 entries, consistency is NOT guaranteed!",
                      Warning)
    print(f'Covariance Fourier Diagonal: {np.real(np.diag(exact_cov_fourier_basis))}')
    print(f'Power spectrum: {power_spectrum}')
    tri_diagonal = np.array([tri_spectrum[i, i, i] for i in range(signal_length)])

    # Optimization step
    g = perform_optimization(tri_spectrum, power_spectrum, signal_length, data_type)
    diagonals: Matrix = extract_diagonals(g, signal_length)
    estimated_covariance: Matrix = construct_estimator(diagonals, power_spectrum, signal_length, noise_power)
    return estimated_covariance


def create_optimization_objective(tri_spectrum, power_spectrum, data_type) -> Callable:
    g_zero = np.outer(power_spectrum, power_spectrum)
    signal_length = len(power_spectrum)

    def objective(matrices_list):
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
                        next_index: int = (k1 + k2 + m) % signal_length
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
    symbolic_matrices = [cp.Variable((signal_length, signal_length), PSD=True, name=f'G{i + 1}')
                         for i in range(signal_length - 1)]
    problem = cp.Problem(cp.Minimize(optimization_objective(symbolic_matrices)), [])

    min_fit_score = problem.solve(solver=cp.CVXOPT)
    optimization_result = [mat.value for mat in symbolic_matrices]

    print(f'Min score: {min_fit_score}')
    print(f'Are matrices None? {[(mat is None) for mat in optimization_result]}')
    print(f'Are matrices Hermitian? {[np.allclose(np.conj(mat), mat) for mat in optimization_result]}')
    print(f'Are matrices PSD? {[np.all(np.linalg.eigvalsh(mat) >= 0) for mat in optimization_result]}')

    return np.array(optimization_result, order='F')


def phase_retrieval(covariance_estimator: Matrix, signal_length: int, approximation_rank: Union[int, None]) -> Matrix:
    """
    The first stage algorithm fot the covariance estimation, see Algorithm 2 in the paper. This stage solves the
    phase ambiguity (up to the inherent ambiguity of the problem).

    Args:
        covariance_estimator(Matrix): The estimator which is given as the output of the first stage of the algorithm
            (Algorithm 1 in the paper).
        signal_length(int): The length of the signal.
        approximation_rank(int): The rank of the approximated covariance matrix.

    Returns:
        The estimated covariance matrix, up to the inherent ambiguity of the problem

    """

    # Building the coefficients matrix A.
    matrix_a: Matrix = coefficient_matrix_construction(covariance_estimator, signal_length, approximation_rank)
    # Finding the singular vector which corresponds to the smallest singular-value of A.
    b = np.dot(np.conj(matrix_a).T, matrix_a)
    val, v = eigh(b, overwrite_a=False, overwrite_b=False, check_finite=False, eigvals=(0, 2))
    if abs(val[1]) < 1e-15:
        warnings.warn("The dimension of the null-space of A is larger the one, so this estimator " +
                      "is NOT guaranteed to succeed!", Warning)

    # Finding the matching angles
    angles = estimate_angles(v, signal_length)
    phases: Vector = np.exp(-1j * angles)
    phases = np.insert(phases, 0, 1)

    # Multiplying by the negative phases to find the Fourier-basis covariance
    covariance_estimator = np.multiply(covariance_estimator, circulant(phases))

    # Converting the estimator back to the standard basis
    covariance_estimator = np.conj(ifft(covariance_estimator, axis=0, norm="ortho").T)
    covariance_estimator = np.conj(ifft(covariance_estimator, axis=0, norm="ortho").T)
    return covariance_estimator


def coefficient_matrix_construction(covariance_estimator: Matrix, signal_length: int,
                                    approximation_rank: Union[int, None]) -> Matrix:
    """
    This function constructs a linear equations system, for solving the phase ambiguity
    (up to the inherent ambiguity of the problem).

    Args:
        covariance_estimator(Matrix): The estimator which is given as the output of the first stage of the algorithm
            (Algorithm 1 in the paper).
        signal_length(int): The length of the signal.
        approximation_rank(int): The rank of the approximated covariance matrix.

    Returns:
        A signal_length ** 3 X (1 + approximation_rank ** 4) matrix (if approximation_rank < sqrt(signal_length))

    """
    rank_sqr: int = approximation_rank ** 2 if approximation_rank is not None else signal_length - 1
    least_eigenvalue_index: int = max(signal_length - rank_sqr, 0)
    last_index: int = signal_length ** 2
    transition_index: int = last_index - signal_length

    hi_i: Matrix = np.power(np.abs(covariance_estimator), 2)
    rotated_mat: Matrix = covariance_estimator
    all_eigenvectors: Matrix = np.empty((signal_length, signal_length, signal_length - least_eigenvalue_index),
                                        order="F", dtype=covariance_estimator.dtype)
    all_m_mats: Matrix = np.zeros((signal_length ** 3, signal_length), dtype=covariance_estimator.dtype)
    sampled_indices: Matrix = np.empty((signal_length, signal_length), dtype=int)
    sampled_indices[0] = np.arange(0, last_index, signal_length + 1)
    for m in range(1, signal_length):
        sampled_hi_indices = np.arange(m, transition_index, signal_length + 1)
        sampled_indices[m] = np.hstack((sampled_hi_indices, np.arange(transition_index,
                                                                      last_index, signal_length + 1)))
        transition_index -= signal_length

    for i in range(signal_length):
        all_eigenvectors[i] = eigh(hi_i, overwrite_b=False, check_finite=False,
                                   eigvals=(least_eigenvalue_index, signal_length - 1))[1]
        rotated_mat = np.roll(rotated_mat, shift=1, axis=1)
        hi_i = np.multiply(covariance_estimator, np.conj(rotated_mat))
        hi_i_plus_one_flat = hi_i.flatten('F')

        for m in range(signal_length):
            all_m_mats[sampled_indices[m] + i * last_index, m] = hi_i_plus_one_flat[sampled_indices[m]]

        rotated_mat = np.roll(rotated_mat, shift=1, axis=0)
        hi_i = np.multiply(covariance_estimator, np.conj(rotated_mat))

    matrix_a = np.hstack((all_m_mats, -block_diag(*vectorized_kron(all_eigenvectors))))
    return matrix_a


def estimate_angles(eigenvector: Vector, signal_length: int) -> Vector:
    """
    This function constructs a linear equations system, for solving the phase ambiguity
    (up to the inherent ambiguity of the problem).

    Args:
        eigenvector(Vector): The eigenvector of the equations matrix which corresponds to the singular value of 0.
        signal_length(int): The length of the signal.

    Returns:
        A vector of estimated phases.

    """
    v = eigenvector[:, 0][:signal_length]
    arguments_vector = np.angle(v)
    angles = np.cumsum(arguments_vector[1:])
    angles = -angles + (angles[signal_length - 2] + arguments_vector[0]) / signal_length * \
        np.arange(1, signal_length, 1)
    return angles
