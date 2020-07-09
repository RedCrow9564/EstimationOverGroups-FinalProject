import numpy as np
from numpy.linalg import norm, qr
from Infrastructure.enums import DistributionType, DistributionParams
from Infrastructure.utils import List, Dict, Scalar, Vector, Matrix


def create_discrete_distribution(distribution_type: str, distribution_length: int, distribution_params: Dict) -> Vector:
    if distribution_type == DistributionType.Uniform:
        return np.ones(distribution_length) / distribution_length
    elif distribution_type == DistributionType.Dirac:
        delta_locations: Vector = np.array(distribution_params[DistributionParams.DeltaLocations])
        distribution: Vector = np.zeros(distribution_length, dtype=np.float64)
        distribution[delta_locations] = 1
        distribution /= len(delta_locations)
        return distribution


def generate_covariance(signal_length: int, approximation_rank: int, data_type, random_generator) -> \
        (Matrix, Matrix, Vector):
    # Picking random eigenvalues (variances) for the exact covariance matrix.
    eigenvalues: Vector = random_generator.uniform(size=approximation_rank)
    eigenvalues /= norm(eigenvalues, ord=1)
    # Creating the set of orthonormal eigenvectors
    eigenvectors: Matrix = random_generator.uniform(size=(signal_length, approximation_rank))
    if data_type in [np.complex128, np.complex64, np.complex]:
        eigenvectors = eigenvectors.astype(data_type)
        eigenvectors += 1j * random_generator.uniform(size=(signal_length, approximation_rank))
    eigenvectors = qr(eigenvectors)[0]
    covariance: Matrix = (eigenvectors * eigenvalues).dot(np.conj(eigenvectors.T))
    return covariance, eigenvectors, eigenvalues


def generate_observations(eigenvectors: Matrix, eigenvalues: Vector, approximation_rank: int, observation_num: int,
                          data_type, random_generator) -> Matrix:

    # Sampling the signal from this covariance matrix.
    standard_deviations: Matrix = np.tile(np.sqrt(eigenvalues).reshape((-1, 1)), (1, observation_num))
    observations: Matrix = random_generator.normal(scale=standard_deviations, size=(approximation_rank, observation_num))
    if data_type in [np.complex128, np.complex64, np.complex]:
        standard_deviations /= np.sqrt(2)
        observations = np.sqrt(0.5) * observations.astype(data_type)
        observations += 1j * random_generator.normal(scale=standard_deviations, size=observations.shape)
    observations = np.dot(eigenvectors, observations).T
    return observations


def generate_shifts_and_noise(observations: Matrix, shifts: List[int], noise_power: float,
                              data_type, random_generator) -> Matrix:
    for i, shift in enumerate(shifts):
        observations[i] = np.roll(observations[i], shift)
    if data_type in [np.complex128, np.complex64, np.complex]:
        half_power_std: Scalar = np.sqrt(noise_power / 2)
        observations += random_generator.normal(loc=0, scale=half_power_std, size=observations.shape)
        observations += 1j * random_generator.normal(loc=0, scale=half_power_std, size=observations.shape)
    else:
        observations += random_generator.normal(loc=0, scale=np.sqrt(noise_power), size=observations.shape)
    return observations
