#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 15.04.2021


'''The `fpi` package implements the Feature Point Identification (FPI)
algorithm for robust voxel-based multiple tracer tracking, using the original
code kindly shared by [1]_.

Summary
-------
A typical workflow for using the `fpi` subpackage would be:

1. Read the LoRs into a `pept.LineData` class instance and set the
   `sample_size` and `overlap` appropriately.
2. Voxellise the `pept.LineData` samples with a `pept.VoxelData` class (this
   can be done on demand, saving memory).
3. Instantiate a `pept.tracking.fpi.FPI` class and transform the voxellised
   LoRs into tracer locations using the `fit` method.

Extended Summary
----------------
[TODO: add more detailed summary of how FPI works].

It was successfully used to track fast-moving radioactive tracers in pipe flows
at the Virginia Commonwealth University. If you use this algorithm in your
work, please cite the original paper [1]_.

Modules Provided
----------------

::

    pept.tracking.fpi
    │
    Functions imported into the subpackage root:
    ├── fpi_ext :   Low-level C++ FPI subroutine.
    │
    Classes imported into the subpackage root:
    └── FPI :       Find tracer locations from samples of voxellised LoRs.

References
----------
.. [1] Wiggins C, Santos R, Ruggles A. A feature point identification method
   for positron emission particle tracking with multiple tracers. Nuclear
   Instruments and Methods in Physics Research Section A: Accelerators,
   Spectrometers, Detectors and Associated Equipment. 2017 Jan 21;843:22-8.
'''


from    .fpi        import  FPI
from    .fpi_ext    import  fpi_ext


__all__ = [
    "FPI",
    "fpi_ext",
]


__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Beta"
