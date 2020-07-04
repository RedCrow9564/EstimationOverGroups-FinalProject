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
from Infrastructure.utils import ex, DataLog, Union, List, Scalar, Vector, Matrix
from Infrastructure.enums import LogFields, DistributionType
from data_generation import create_discrete_distribution, generate_signal, generate_observations
from covariance_estimation import low_rank_multi_reference_factor_analysis


def calc_estimation_error(exact_covariance, estimated_covariance):
    covariance_norm: Scalar = norm(exact_covariance, ord='fro') ** 2
    rotated_cov: Matrix = exact_covariance
    error: Scalar = norm(estimated_covariance - rotated_cov, ord='fro') ** 2

    for _ in range(exact_covariance.shape[0]):
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
    observations_numbers: List[int] = [8]
    approximation_ranks: List[Union[int, None]] = [2]
    noise_powers: List[float] = [0.0]
    trials_num: int = 1
    first_seed: int = 200
    shifts_distribution_type = DistributionType.Uniform
    experiment_name: str = "Testing Code Infrastructure"
    results_path: str = r'Results/'


@ex.automain
def main(signal_lengths: List[int], observations_numbers: List[int], approximation_ranks: List[int],
         noise_powers: List[float], shifts_distribution_type: str, trials_num: int, data_type, results_path: str,
         experiment_name: str, first_seed: int) -> None:
    """ The main function of this project

    This functions performs the desired experiment according to the given configuration.
    The function runs the random_svd and random_id for every combination of data_size, approximation rank and increment
    given in the config and saves all the results to a csv file in the results folder (given in the configuration).
    """
    results_log = DataLog(LogFields)  # Initializing an empty results log.

    for signal_length, noise_power, approximation_rank, observations_num in product(
            signal_lengths, noise_powers, approximation_ranks, observations_numbers):
        shifts_distribution: List[float] = create_discrete_distribution(shifts_distribution_type, signal_length)

        mean_error: float = 0
        trials_seeds: Vector = np.arange(first_seed, first_seed + trials_num).tolist()
        for trial_seed in trials_seeds:
            rng = Generator(PCG64(trial_seed))  # Set trial's random generator.
            exact_signal, exact_covariance = generate_signal(signal_length, approximation_rank, data_type, rng)
            observations_shifts: List[int] = rng.choice(signal_length, size=observations_num, p=shifts_distribution)
            observations: Matrix = generate_observations(exact_signal, observations_num, observations_shifts,
                                                         noise_power, data_type, rng)
            observations_fourier: Matrix = fft(observations, axis=1)
            # TODO: Remove the exact_covariance argument for real experiments.
            estimated_covariance: Matrix = low_rank_multi_reference_factor_analysis(
                observations_fourier, signal_length, approximation_rank, noise_power, data_type, exact_covariance)
            mean_error += calc_estimation_error(exact_covariance, estimated_covariance)

        mean_error /= trials_num
        print(f'Finished experiment of signal length L={signal_length}, r={approximation_rank}, noise={noise_power} ' +
              f'with error {mean_error}')

        # Appending all the experiment results to the log.
        results_log.append(LogFields.DataSize, signal_length)
        results_log.append(LogFields.DataType, data_type.__name__)
        results_log.append(LogFields.ApproximationRank, approximation_rank)
        results_log.append(LogFields.NoisePower, noise_power)
        results_log.append(LogFields.ObservationsNumber, observations_num)
        results_log.append(LogFields.TrialsNum, trials_num)
        results_log.append(LogFields.ShiftsDistribution, shifts_distribution_type)
        results_log.append(LogFields.MeanError, mean_error)

    results_log.save_log(f'{experiment_name} results', results_folder_path=results_path)
