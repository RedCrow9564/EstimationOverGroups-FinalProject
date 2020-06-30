import numpy as np
from numpy.linalg import norm, qr
from Infrastructure.enums import DistributionType
from Infrastructure.utils import List, Scalar, Vector, Matrix


def create_discrete_distribution(distribution_type: str, distribution_length: int, **kwargs) -> List[float]:
    if distribution_type == DistributionType.Uniform:
        return np.ones(distribution_length) / distribution_length
    elif distribution_type == DistributionType.Dirac:
        delta_location: int = kwargs["delta_location"]
        distribution: List[float] = np.zeros(distribution_length).tolist()
        distribution[delta_location] = 1
        return distribution


def generate_signal(signal_length: int, approximation_rank: int, data_type, random_generator) -> (Vector, Matrix):
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

    # Sampling the signal from this covariance matrix.
    signal: Vector = np.multiply(eigenvalues, random_generator.standard_normal(size=approximation_rank))
    signal = eigenvectors.dot(signal)
    return signal, covariance


def generate_observations(signal: Matrix, observations_num: int, shifts: List[int], noise_power: float,
                          data_type, random_generator) -> Matrix:
    observations = np.tile(signal, (observations_num, 1))
    for i, shift in enumerate(shifts):
        observations[i] = np.roll(observations[i], shift)
    if data_type in [np.complex128, np.complex64, np.complex]:
        half_power_std: Scalar = np.sqrt(noise_power / 2)
        observations += random_generator.normal(loc=0, scale=half_power_std, size=observations.shape)
        observations += 1j * random_generator.normal(loc=0, scale=half_power_std, size=observations.shape)
    else:
        observations += random_generator.normal(loc=0, scale=np.sqrt(noise_power), size=observations.shape)
    return observations
