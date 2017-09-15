#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:10:00 2017

@author: michael
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy   # << New line

ext_modules = [Extension("ea_util", ["ea_util.pyx"])]

setup(
    name='ea_util',
    cmdclass={'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    ext_modules=ext_modules
)
