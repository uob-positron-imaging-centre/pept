#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.08.2019


'''PEPT-oriented visualisation tools.

Summary
-------
Visualisation functions and classes for PEPT data, transparently working with
both `pept` base classes and raw NumPy arrays (e.g. `PlotlyGrapher.add_lines`
handles both `pept.LineData` and (N, 7) NumPy arrays).

The `PlotlyGrapher` class creates interactive, publication-ready 3D figures
with optional subplots which can also be exported to portable HTML files. The
`PlotlyGrapher2D` class is its two-dimensional counterpart, handling e.g.
`pept.Pixels`.

Modules Provided
----------------

::

    pept.visualisation
    │
    Classes imported into the subpackage root:
    ├── PlotlyGrapher :     Plotly-based interactive 3D graphs with subplots.
    ├── PlotlyGrapher2D :   Plotly-based interactive 2D graphs with subplots.
    │
    Modules provided:
    └── plotly_grapher

'''


from    .plotly_grapher     import  PlotlyGrapher
from    .plotly_grapher2d   import  PlotlyGrapher2D


__all__ = [
    "PlotlyGrapher",
    "PlotlyGrapher2D",
]


__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Beta"
