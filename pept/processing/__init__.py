#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.06.2020


'''The PEPT-oriented post-processing suite, including occupancy grid,
vector velocity fields, etc.

This module contains fast, robust functions that operate on PEPT-like data
and integrate with the `pept` library's base classes.

'''


from    .occupancy      import  circles2d
from    .occupancy      import  occupancy2d

from    .circles_ext    import  circles2d_ext
from    .occupancy_ext  import  occupancy2d_ext


__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Beta"
