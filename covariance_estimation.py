import numpy as np
from scipy.linalg import eigh, block_diag, circulant
import cvxpy as cp
from itertools import product
import warnings
from Infrastructure.utils import Union, List, Vector, Matrix, ThreeDMatrix, Callable
from polyspectra_estimation import estimate_power_spectrum, estimate_tri_spectrum_v2
from vectorized_actions import extract_diagonals, construct_estimator, vectorized_kron, \
    change_from_fourier_basis


def low_rank_multi_reference_factor_analysis(
        observations_fourier: Matrix, signal_length: int, approximation_rank: Union[int, None],
        noise_power: float, data_type) -> Matrix:
    """
    The entire algorithm fot the covariance estimation. It consists of two stages, see Algorithm 1 and Algorithm 2 in
    the paper.

    Args:
        observations_fourier(Matrix): The fourier coefficients of all observations.
        signal_length(int): The length of the signal.
        approximation_rank(int): The rank of the approximated covariance matrix.
        noise_power(float): An estimator for the noise power in the sampled observations.
        data_type: Either np.float64 for real-data or np.complex128 for complex data.

    Returns:
        The estimated covariance matrix.

    """
    estimator = estimate_covariance_up_to_phases(observations_fourier, signal_length, noise_power, data_type)
    estimator = phase_retrieval(estimator, signal_length, approximation_rank)
    return estimator


def estimate_covariance_up_to_phases(observations_fourier: Matrix, signal_length: int, noise_power: float,
                                     data_type) -> Matrix:
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
    print("Started covariance estimation up to phase ambiguity (Algorithm 1)")
    power_spectrum: Vector = estimate_power_spectrum(observations_fourier)
    tri_spectrum: ThreeDMatrix = estimate_tri_spectrum_v2(observations_fourier)

    # Optimization step
    g, _ = perform_optimization(tri_spectrum, power_spectrum, signal_length, data_type)

    # Construct the estimator Cx.
    diagonals: Matrix = extract_diagonals(g, signal_length)
    estimated_covariance: Matrix = construct_estimator(diagonals, power_spectrum, signal_length, noise_power)
    return estimated_covariance


def create_optimization_objective(tri_spectrum: ThreeDMatrix, power_spectrum: Vector, data_type,
                                  use_cp: bool = True) -> (Callable, List[Matrix], List):
    """
    The function creates the optimization objective function, using the estimations of the tri-spectrum,
    and the power-spectrum. The objective function depends on whether the data is sampled for real distributions
    or complex distributions.

    Args:
        tri_spectrum(ThreeDMatrix): An estimation of the signal's tri-spectrum.
        power_spectrum(Vector): An estimation of the signal's power-spectrum.
        data_type: Either np.float64 for real-data or np.complex128 for complex data.
        use_cp(bool): A flag which indicates if the returned output of the objective should be symbolic
            (used for the CVXPY optimization) or numeric (used for unit-testing).

    Returns:
        A list of matrices which minimize the optimization objective and the minimal fit error.

    """
    signal_length = tri_spectrum.shape[0]
    symbolic_matrices = [cp.Variable((signal_length, signal_length), complex=True, name=f'G{i}')
                         for i in range(signal_length)]
    constraints = [G >> 0 for G in symbolic_matrices[1:]]
    constraints.extend([symbolic_matrices[0] == np.outer(power_spectrum, power_spectrum)])
    
    
    def objective(matrices_list):
        error_terms: Vector = []
        is_data_real: bool = data_type in [np.float32, np.float64]
        for k1, k2, m in product(range(signal_length), repeat=3):
            other_index = (k2 - k1) % signal_length
            k1_plus_m = (k1 + m) % signal_length
            current_term = tri_spectrum[k1, k1_plus_m, (k2 + m) % signal_length]
            current_term -= matrices_list[other_index][k1, k1_plus_m]
            current_term -= matrices_list[m][k1, k2]

            if is_data_real:
                next_index: int = (k1 + k2 + m) % signal_length
                current_term -= matrices_list[next_index][
                    - k2 % signal_length, (-k2 - m) % signal_length]

            error_terms.append(current_term)

        if use_cp:
            fit_score = cp.sum([cp.abs(term) ** 2 for term in error_terms])
        else:
            fit_score = np.sum([np.abs(term) ** 2 for term in error_terms])
        return fit_score
    return objective, symbolic_matrices, constraints


def perform_optimization(tri_spectrum: ThreeDMatrix, power_spectrum: Matrix, signal_length: int,
                         data_type) -> (ThreeDMatrix, float):
    """
    The optimization step of Algorithm 1 of the paper. This step utilizes the CVXPY library
    to fit (L - 1) matrices of size L x L which are all hermitian and PSD. This function returns
    the fitted matrices and the minimal error.

    Args:
        tri_spectrum(ThreeDMatrix): An estimation of the signal's tri-spectrum.
        power_spectrum(Matrix): An estimation of the signal's power-spectrum.
        signal_length(int): The length of the signal.
        data_type: Either np.float64 for real-data or np.complex128 for complex data.

    Returns:
        A list of matrices which minimize the optimization objective and the minimal fit error.

    """
    optimization_objective, symbolic_matrices, constraints = create_optimization_objective(
        tri_spectrum, power_spectrum, data_type)

    objective = optimization_objective(symbolic_matrices)
    problem = cp.Problem(cp.Minimize(objective), constraints)

    # Solving the convex optimization problem and gathering the fitted matrices.
    min_fit_score = problem.solve(solver=cp.SCS, eps=1e-25, warm_start=False)
    optimization_result = [mat.value for mat in symbolic_matrices[1:]]
    print(f'Min fit score: {min_fit_score}')

    return np.array(optimization_result, order='F'), min_fit_score


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

    print("Started phase-retrieval step (Algorithm 2)")
    # Building the coefficients matrix A.
    matrix_a: Matrix = coefficient_matrix_construction(covariance_estimator, signal_length, approximation_rank)
    # Finding the singular vector which corresponds to the smallest singular-value of A.
    b = np.dot(np.conj(matrix_a).T, matrix_a)
    val, v = eigh(b, overwrite_a=False, overwrite_b=True, check_finite=False, eigvals=(0, 2))
    if abs(val[1]) < 1e-15:
        warnings.warn("The dimension of the null-space of A is larger the one, so this estimator " +
                      "is NOT guaranteed to succeed!", Warning)

    # Finding the matching angles
    v = np.roll(v[:, 0][:signal_length], shift=1)
    angles = estimate_angles(v, signal_length)
    phases: Vector = np.exp(-1j * angles)
    phases = np.insert(phases, 0, 1)

    # Multiplying by the negative phases to find the Fourier-basis covariance
    covariance_estimator = np.multiply(covariance_estimator, circulant(phases))

    # Converting the estimator back to the standard basis
    covariance_estimator = change_from_fourier_basis(covariance_estimator)
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

    matrix_a = np.hstack((all_m_mats, block_diag(*-vectorized_kron(all_eigenvectors))))
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
    v = eigenvector[:signal_length]
    arguments_vector = np.angle(v)
    args_cumsum: Vector = np.cumsum(arguments_vector[1:])
    total_sum: float = args_cumsum[signal_length - 2] + arguments_vector[0]
    total_sum /= signal_length
    angles = args_cumsum - total_sum * np.arange(1, signal_length)
    return angles
        
