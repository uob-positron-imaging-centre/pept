#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.08.2019


'''The *plotly_grapher* module implements Plotly-based visualisation tools
to aid PEPT data analysis and to create publication-ready figures.

The *PlotlyGrapher* class can create and automatically configure 3D subplots
for PEPT data visualisation. The PEPT 3D axes convention has the *y*-axis
pointing upwards, such that the vertical screens of a PEPT scanner represent
the *xy*-plane. The class provides functionality for plotting 3D scatter or
line plots, with optional colorbars.

If you use the `pept` package, we ask you to cite the following paper:

    Nicuşan AL, Windows-Yule CR. Positron emission particle tracking
    using machine learning. Review of Scientific Instruments.
    2020 Jan 1;91(1):013329.
    https://doi.org/10.1063/1.5129251

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


