#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.08.2019


'''PEPT-oriented visualisation tools.

Summary
-------
This subpackage hosts visualisation functions and classes for PEPT data. At the
moment, there is only one module, `plotly_grapher`, which implements
Plotly-based visualisation tools to aid PEPT data analysis, creating 3D
interactive, publication-ready figures.

Modules Provided
----------------

::

    pept.visualisation
    │
    Classes imported into the subpackage root:
    ├── PlotlyGrapher :     Plotly-based interactive 3D graphs with subplots.
    │
    Modules provided:
    └── plotly_grapher

'''


from    .plotly_grapher     import  PlotlyGrapher


__all__ = [
    "PlotlyGrapher"
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
__status__ = "Stable"


