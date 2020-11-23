'''
The Cython build script.  Called from Makefile
'''
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension("pycarfac", sources=["pycarfac.pyx", "../ccarfac/carfac.c"])
setup(name="pycarfac", ext_modules=cythonize([ext]),
        include_dirs=[np.get_include(), "../ccarfac/"])

