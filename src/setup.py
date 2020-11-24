'''
The Cython build script.  Called from Makefile
'''
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

ccarfacloc = os.path.join(os.curdir, "..", "ccarfac", "src")

ext = Extension("pycarfac", sources=["pycarfac.pyx", os.path.join(ccarfacloc, "carfac.c")])
setup(name="pycarfac", ext_modules=cythonize([ext]),
        include_dirs=[np.get_include(), ccarfacloc])

