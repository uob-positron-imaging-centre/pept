#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.06.2020


'''The PEPT-oriented post-processing suite, including occupancy grid,
vector velocity fields, etc.

Summary
-------
This module contains fast, robust functions that operate on PEPT-like data
and integrate with the `pept` library's base classes.

Modules Provided
----------------

::

    pept.processing
    │
    Functions imported into the subpackage root:
    ├── occupancy2d :           Pixellised occupancy grid from points.
    └── occupancy2d_ext :       Occupancy grid low-level Cython extension.

'''


from    .occupancy      import  occupancy2d
from    .occupancy_ext  import  occupancy2d_ext


__all__ = [
    "occupancy2d",
    "occupancy2d_ext"
]


__author__ = "Andrei Leonard Nicusan"
__credits__ = [
    "Andrei Leonard Nicusan",
    "Kit Windows-Yule",
    "Sam Manger"
]
__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Development"


