#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
main.py - The main module of the project
========================================

This module contains the config for the experiment in the "config" function.
Running this module invokes the :func:`main` function, which then performs the experiment and saves its results
to the configured results folder. Example for running an experiment: ``python main.py``

"""
import numpy as np
from numpy import roll
from numpy.linalg import norm
from numpy.random import Generator, PCG64
from numpy.fft import fft
from itertools import product
import warnings
from Infrastructure.utils import ex, DataLog, Dict, Union, List, Scalar, Vector, Matrix
from Infrastructure.enums import LogFields, DistributionType, DistributionParams
from data_generation import create_discrete_distribution, generate_covariance, generate_observations, \
    generate_shifts_and_noise
from vectorized_actions import change_to_fourier_basis
from covariance_estimation import low_rank_multi_reference_factor_analysis


def calc_estimation_error(exact_covariance, estimated_covariance):
    """
    This function measures the error of the estimator, given the covariance exact value.

    Args:
        exact_covariance(Matrix): The exact covariance (square) matrix in Fourier basis.
        estimated_covariance(Matrix): The estimation for the exact covariance.

    Returns:
        The estimation error.
    """
    covariance_norm: Scalar = norm(exact_covariance, ord='fro') ** 2
    rotated_cov: Matrix = exact_covariance
    error: Scalar = norm(estimated_covariance - rotated_cov, ord='fro') ** 2

    for _ in range(exact_covariance.shape[0] - 1):
        rotated_cov = roll(rotated_cov, shift=[1, 1], axis=[0, 1])
        shifted_error = norm(estimated_covariance - rotated_cov, ord='fro') ** 2
        if shifted_error < error:
            error = shifted_error

    return error / covariance_norm


@ex.config
def config():
    """ Config section

    This function contains all possible configuration for all experiments. Full details on each configuration values
    can be found in :mod:`enums.py`.
    """

    data_type = np.complex128
    signal_lengths: [int] = [5]
    observations_numbers: List[int] = [100000]
    approximation_ranks: List[Union[int, None]] = [2]
    noise_powers: List[float] = [0.0]
    trials_num: int = 10
    first_seed: int = 200
    shifts_distribution_type = DistributionType.Uniform
    distribution_params: Dict = {
        DistributionParams.DeltaLocations: [1]
    }
    experiment_name: str = "Test2"
    results_path: str = r'Results/'


@ex.automain
def main(signal_lengths: List[int], observations_numbers: List[int], approximation_ranks: List[int],
         noise_powers: List[float], shifts_distribution_type: str, trials_num: int, data_type, results_path: str,
         experiment_name: str, first_seed: int, distribution_params: Dict) -> None:
    """ The main function of this project

    This functions performs the desired experiment according to the given configuration.
    The function runs the random_svd and random_id for every combination of data_size, approximation rank and increment
    given in the config and saves all the results to a csv file in the results folder (given in the configuration).
    """
    results_log = DataLog(LogFields)  # Initializing an empty results log.

    for signal_length, noise_power, approximation_rank, observations_num in product(
            signal_lengths, noise_powers, approximation_ranks, observations_numbers):
        if approximation_rank >= np.sqrt(signal_length):
            warnings.warn(f'Approximation rank {approximation_rank} is at least the square-root of the ' +
                          f'signal length {signal_length}, consistency is NOT guaranteed!', Warning)
        shifts_distribution: Vector = create_discrete_distribution(shifts_distribution_type, signal_length,
                                                                   distribution_params)
        mean_error: float = 0
        max_error: float = 0.0
        trials_seeds: Vector = np.arange(first_seed, first_seed + trials_num).tolist()

        for trial_index, trial_seed in enumerate(trials_seeds):
            print(f'Started trial no. {trial_index}')
            rng = Generator(PCG64(trial_seed))  # Set trial's random generator.
            exact_covariance, eigenvectors, eigenvalues = generate_covariance(signal_length, approximation_rank,
                                                                              data_type, rng)
            observations = generate_observations(eigenvectors, eigenvalues, approximation_rank, observations_num,
                                                 data_type, rng)
            observations_shifts: List[int] = rng.choice(signal_length, size=observations_num, p=shifts_distribution)
            observations = generate_shifts_and_noise(observations, observations_shifts, noise_power, data_type, rng)
            observations_fourier: Matrix = fft(observations, norm="ortho", axis=1)

            exact_cov_fourier_basis: Matrix = change_to_fourier_basis(exact_covariance)
            if np.any(np.abs(exact_cov_fourier_basis)) < 1e-15:
                warnings.warn("The covariance matrix in Fourier basis has some 0 entries, " +
                              "consistency is NOT guaranteed!", Warning)

            estimated_covariance: Matrix = low_rank_multi_reference_factor_analysis(
                observations_fourier, signal_length, approximation_rank, noise_power, data_type)
            current_error: float = calc_estimation_error(exact_covariance, estimated_covariance)
            mean_error += current_error
            print(f'Current error: {current_error}')
            max_error = max(max_error, current_error)

        mean_error /= trials_num
        print(f'Finished experiment of signal length L={signal_length}, n={observations_num}, '
              f'r={approximation_rank}, noise={noise_power} with mean error {mean_error} and max error {max_error}')

        # Appending all the experiment results to the log.
        results_log.append(LogFields.DataSize, signal_length)
        results_log.append(LogFields.DataType, data_type.__name__)
        results_log.append(LogFields.ApproximationRank, approximation_rank)
        results_log.append(LogFields.NoisePower, noise_power)
        results_log.append(LogFields.ObservationsNumber, observations_num)
        results_log.append(LogFields.TrialsNum, trials_num)
        results_log.append(LogFields.ShiftsDistribution, shifts_distribution_type)
        results_log.append(LogFields.MeanError, mean_error)
        results_log.append(LogFields.MaxError, max_error)

    results_log.save_log(f'{experiment_name} results', results_folder_path=results_path)
