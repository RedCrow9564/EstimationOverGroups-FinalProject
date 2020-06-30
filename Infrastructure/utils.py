# -*- coding: utf-8 -*-
"""
utils.py - The common utilities functions and objects
=====================================================

This module contains all frequently-used methods and objects which can be shared among the entire project.
For example, data types name used for type-hinting, a basic enum class :class:`BaseEnum`, methods for measuring
run-time of a given function.
"""
from typing import List, Dict, Callable, Union, Iterator, Tuple
from nptyping import Array
import inspect
import time
import pandas as pd
import os
from Infrastructure import pyximportcpp
import numpy
from sacred import Experiment

# Defining the "sacred" experiment object.
ex = Experiment(name="Initializing project", interactive=False)
pyximportcpp.install(setup_args={"include_dirs": numpy.get_include()}, reload_support=True)

# Naming data types for type hinting.
Number = Union[int, complex]
Scalar = Union[Number, Array[complex, 1, 1], Array[int, 1, 1]]
RowVector = Union[List[Scalar], Array[float, 1, ...], Array[int, 1, ...], Scalar]
ColumnVector = Union[List[Scalar], Array[float, ..., 1], Array[int, ..., 1], Scalar]
Vector = Union[RowVector, ColumnVector]
Matrix = Union[List[Vector], Array[float], Array[int], Vector, Scalar]


class _MetaEnum(type):
    """
    A private meta-class which given any :class:`BaseEnum` object to be an iterable.
    This can be used for iterating all possible values of this enum. Should not be used explicitly.
    """
    def __iter__(self) -> Iterator:
        """
        This method gives any BaseEnum the ability of iterating over all the enum's values.

        Returns:
            An iterator for the collection of all the enum's values.

        """
        # noinspection PyUnresolvedReferences
        return self.enum_iter()

    def __contains__(self, item) -> bool:
        """
        This method give any BaseEnum the ability to test if a given item is a possible value for this enum class.

        Returns:
            A flag which indicates if 'item' is a possible value for this enum class.

        """
        # noinspection PyUnresolvedReferences
        return self.enum_contains(item)


class BaseEnum(metaclass=_MetaEnum):
    """
    A basic interface for all enum classes. Should be sub-classed in eny enum, i.e ``class ExperimentType(BaseEnum)``
    """

    @classmethod
    def enum_iter(cls) -> Iterator:
        """
        This method gives any BaseEnum the ability of iterating over all the enum's values.

        Returns:
            An iterator for the collection of all the enum's values.

        """
        return iter(cls.get_all_values())

    @classmethod
    def enum_contains(cls, item) -> bool:
        """
        This method give any BaseEnum the ability to test if a given item is a possible value for this enum class.

        Returns:
            A flag which indicates if 'item' is a possible value for this enum class.

        """
        return item in cls.get_all_values()

    @classmethod
    def get_all_values(cls) -> List:
        """
        A method which fetches all possible values of an enum. Used for iterating over an enum.

        Returns:
            A list of all possible enum's values.

        """
        all_attributes: List = inspect.getmembers(cls, lambda a: not inspect.ismethod(a))
        all_attributes = [value for name, value in all_attributes if not (name.startswith('__') or name.endswith('__'))]
        return all_attributes


def create_factory(possibilities_dict: Dict[str, Callable], are_methods: bool = False) -> Callable:
    """
    A generic method for creating factories for the entire project.

    Args:
        possibilities_dict(Dict[str, Callable]): The dictionary which maps object types (as strings!) and returns the
            relevant class constructors.
        are_methods(bool): A flag, true if the factory output are methods, rather than objects. Defaults to False

    Returns:
         The factory function for the given classes/methods mapping.

    """
    def factory_func(requested_object_type: str):  # Inner function!
        if requested_object_type not in possibilities_dict:
            raise ValueError("Object type {0} is NOT supported".format(requested_object_type))
        else:
            if are_methods:
                return possibilities_dict[requested_object_type]
            else:
                return possibilities_dict[requested_object_type]()

    return factory_func


def measure_time(method: Callable) -> Callable:
    """
    A method which receives a method and returns the same method, while including run-time measure
    output for the given method, in seconds.

    Args:
        method(Callable): A method whose run-time we are interested in measuring.

    Returns:
        A function which does exactly the same, with an additional run-time output value in seconds.

    """
    def timed(*args, **kw) -> Tuple:
        ts = time.perf_counter_ns()
        result = method(*args, **kw)
        te = time.perf_counter_ns()
        duration_in_ms: float = (te - ts) * 1e-9
        if isinstance(result, tuple):
            return result + (duration_in_ms,)
        else:
            return result, duration_in_ms
    timed.__name__ = method.__name__ + " with time measure"
    return timed


def is_empty(collection: List) -> bool:
    return len(collection) == 0


class DataLog:
    """
    A class for log-management objects. See the following example for creating it: ``DataLog(["Column 1", "Column 2"])``
    """
    def __init__(self, log_fields: List):
        """
        This methods initializes an empty log.

        Args:
            log_fields(List) - A list of column names for this log.

        """
        self._data: Dict = dict()
        self._log_fields: List = log_fields
        for log_field in log_fields:
            self._data[log_field] = list()

    def append(self, data_type: str, value: Scalar) -> None:
        """
        This methods appends given data to the given column inside the log.
        Example of usage:``log.append(DataFields.DataSize, 20)``

        Args:
            data_type(LogFields): The column name in which the input data in inserted to.
            value(Scalar): The value to insert to the log.

        """
        self._data[data_type].append(value)

    def append_dict(self, data_dict: Dict) -> None:
        """
        This methods takes the data from the input dictionary and inserts it to this log.

        Args:
            data_dict(Dict): The dictionary from which new data is taken and inserted to the log.

        """
        for log_field, data_value in data_dict.items():
            self.append(log_field, data_value)

    def save_log(self, log_file_name: str, results_folder_path: str) -> None:
        """
        This method saves the log to a file, with the input name, in the input folder path.

        Args:
            log_file_name(str): The name for this log file.
            results_folder_path(str): The path in which this log will be saved.

        """
        df = pd.DataFrame(self._data, columns=self._log_fields)
        df.to_csv(os.path.join(results_folder_path, log_file_name + ".csv"), sep=",", float_format="%.2E", index=False)
        ex.info["Experiment Log"] = self._data
