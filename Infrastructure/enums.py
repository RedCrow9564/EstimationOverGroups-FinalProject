# -*- coding: utf-8 -*-
"""
enums.py - All enums section
============================

This module contains all possible enums of this project. Most of them are used by the configuration section in
:mod:`main`. An example for using enum:
::
    DistributionType.Uniform

"""

from Infrastructure.utils import BaseEnum


class LogFields(BaseEnum):
    """
    The enum class of fields within experiments logs. Possible values:

    * ``LogFields.DataSize``

    * ``LogFields.DataType``

    * ``LogFields.ApproximationRank``

    * ``LogFields.ObservationsNumber``

    * ``LogFields.NoisePower``

    * ``LogFields.TrialsNum``

    * ``LogFields.ShiftsDistribution``

    * ``LogFields.MeanError``
    """
    DataSize: str = "Data size"
    DataType = "Data type (complex/real)"
    ApproximationRank: str = "r"
    ObservationsNumber: str = "Observations Number"
    NoisePower: str = "Noise power"
    TrialsNum: str = "Number of Trials"
    ShiftsDistribution: str = "Shifts Distribution"
    MeanError: str = "Mean Error"


class DistributionType(BaseEnum):
    """
    The enum class of experiment types. Possible values:

    * ``DistributionType.Uniform``

    * ``DistributionType.Dirac``

    """
    Uniform: str = "Uniform Distribution"
    Dirac: str = "Dirac Delta Distribution"
