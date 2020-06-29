# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: License: GNU v3.0
# Author : Sam Manger <s.manger@bham.ac.uk>
# Date   : 20.08.2019


'''The `birmingham_method` package provides an efficient, optionally-parallel
implementation of the well-known Birmingham method for single-tracer tracking.

Summary
-------
A typical workflow for using the `birmingham_method` package would be:

1. Read the LoRs into a `pept.LineData` class instance and set the
   `sample_size` and `overlap` appropriately.
2. Instantiate a `pept.tracking.birmingham_method.BirminghamMethod` class and
   transform the LoRs into tracer locations using the `fit` method.

Extended Summary
----------------
For a given "sample" of LoRs, the Birmingham method minimises the distance
between all of the LoRs, rejecting a fraction of lines that lie furthest away
from the calculated distance. The process is repeated iteratively until a
specified fraction ("fopt") of the original subset of LORs remains.

The Birmingham method has been used extensively for well over 30 years at the
University of Birmingham to track radioactively-labelled tracers in a variety
of industrial and scientific systems [1]_.

Modules Provided
----------------

::

    pept.tracking.birmingham_method
    │
    Classes imported into the subpackage root:
    └── BirminghamMethod :  Transform samples of LoRs into tracer locations.

References
----------
.. [1] Parker DJ, Broadbent CJ, Fowles P, Hawkesworth MR, McNeil P. Positron
   emission particle tracking-a technique for studying flow within engineering
   equipment. Nuclear Instruments and Methods in Physics Research Section A:
   Accelerators, Spectrometers, Detectors and Associated Equipment. 1993 Mar
   10;326(3):592-607.
'''


from    .birmingham_method    import BirminghamMethod


__all__ = [
   'BirminghamMethod'
]


__author__ = "Sam Manger"
__credits__ = [
    "Sam Manger",
    "Andrei Leonard Nicusan",
    "Kit Windows-Yule",
]
__license__ = "GNU v3.0"
__maintainer__ = "Sam Manger"
__email__ = "s.manger@bham.ac.uk"
__status__ = "Development"


