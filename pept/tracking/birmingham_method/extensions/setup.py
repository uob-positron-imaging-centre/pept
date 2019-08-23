#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : setup.py
# License           : License: GNU v3.0
# Author            : Andrei Leonard Nicusan <aln705@student.bham.ac.uk>
# Date              : 01.07.2019


import io
import os

from distutils.core import setup
from Cython.Build   import cythonize

# Package meta-data.
NAME = 'birmingham_method'
DESCRIPTION = 'Given coordinates of a set of lines, find the midpoint using the Birmingham Method for PEPT tracking'
URL = 'https://github.com/spmngr'
EMAIL = 's.manger@bham.ac.uk'
AUTHOR = 'Sam Manger'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

TO_COMPILE = 'birmingham_method.pyx'
COMPILER_DIRECTIVES = {'language_level' : "3"}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,

    ext_modules=cythonize(TO_COMPILE),

    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)

