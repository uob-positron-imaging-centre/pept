#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 19.08.2019


# Import base data structures
from    .data.base  import  LineData
from    .data.base  import  PointData

# Import subpackages
from    .           import  data
from    .           import  scanners
from    .           import  simulation
from    .           import  diagnostics
from    .           import  tracking
from    .           import  visualisation


__all__ = [
    'LineData',
    'PointData',
    'data',
    'scanners',
    'simulation',
    'diagnostics',
    'tracking',
    'visualisation'
]


__author__ =        "Andrei Leonard Nicusan"
__credits__ =       ["Andrei Leonard Nicusan", "Kit Windows-Yule", "Sam Manger"]
__license__ =       "GNU v3.0"
__version__ =       "0.1"
__maintainer__ =    "Andrei Leonard Nicusan"
__email__ =         "a.l.nicusan@bham.ac.uk"
__status__ =        "Development"


