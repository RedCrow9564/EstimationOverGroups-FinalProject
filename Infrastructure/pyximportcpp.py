# -*- coding: utf-8 -*-
"""
pyximportcpp.py - Cython compilation module in C++
==================================================

This module is a short script which guarantees the import of pyximportcpp (instead of pyximport)
results in Cython code compilation to C++ (instead of the default C language).
Credits for this code belong to "user4967717" in "Stack Overflow"
(https://stackoverflow.com/questions/21938065/how-to-configure-pyximport-to-always-make-a-cpp-file).
"""
import pyximport
from pyximport import install

old_get_distutils_extension = pyximport.pyximport.get_distutils_extension

def new_get_distutils_extension(modname, pyxfilename, language_level=None):
    extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
    extension_mod.language='c++'
    return extension_mod,setup_args

pyximport.pyximport.get_distutils_extension = new_get_distutils_extension
