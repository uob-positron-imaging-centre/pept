#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
#
#    If you used this codebase or any software making use of it in a scientific
#    publication, you must cite the following paper:
#        Nicuşan AL, Windows-Yule CR. Positron emission particle tracking
#        using machine learning. Review of Scientific Instruments.
#        2020 Jan 1;91(1):013329.
#        https://doi.org/10.1063/1.5129251
#
#    Copyright (C) 2021 the pept developers
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# File   : parallel_screens.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.08.2019


import  time
import  numpy           as      np
from    scipy.integrate import  quad

from    pept            import  LineData
from    pept.base       import  PEPTObject
from    pept.utilities  import  read_csv




def parallel_screens(
    filepath_or_array,
    screen_separation,
    sample_size = None,
    overlap = None,
    verbose = True,
    **kwargs,
):
    '''Initialise PEPT LoRs for parallel screens PET/PEPT detectors from an
    input CSV file or array.

    **The expected data columns in the file are `[time, x1, y1, x2, y2]`**.
    This is automatically transformed into the standard `Lines` format with
    columns being `[time, x1, y1, z1, x2, y2, z2]`, where `z1 = 0` and
    `z2 = screen_separation`.

    `ParallelScreens` can be initialised with a predefined numpy array of LoRs
    or read data from a `.csv`.

    Parameters
    ----------
    filepath_or_array : [str, pathlib.Path, IO] or numpy.ndarray (N, 5)
        A path to a file to be read from or an array for initialisation. A
        path is a string with the (absolute or relative) path to the data
        file or a URL from which the PEPT data will be read. It should
        include the full file name, along with its extension (.csv, .a01,
        etc.).

    screen_separation : float
        The separation (in *mm*) between the two PEPT screens corresponding
        to the `z` coordinate of the second point defining each line. The
        attribute `lines`, with columns
        `[time, x1, y1, z1, x2, y2, z2]`, will have `z1 = 0` and
        `z2 = screen_separation`.

    sample_size : int, default 0
        An `int` that defines the number of lines that should be returned
        when iterating over `lines`. A `sample_size` of 0 yields all the
        data as one single sample. A good starting value would be 200 times
        the maximum number of tracers that would be tracked.

    overlap : int, default 0
        An `int` that defines the overlap between two consecutive samples
        that are returned when iterating over `lines`. An overlap of 0
        implies consecutive samples, while an overlap of
        (`sample_size` - 1) means incrementing the samples by one. A
        negative overlap means skipping values between samples. An error is
        raised if `overlap` is larger than or equal to `sample_size`.

    verbose : bool, default True
        An option that enables printing the time taken for the
        initialisation of an instance of the class. Useful when reading
        large files (10gb files for PEPT data is not unheard of).

    kwargs : other keyword arguments
        Other keyword arguments to be passed to `pept.read_csv`, e.g.
        "skiprows" or "max_rows". See the `pept.read_csv` documentation for
        other arguments.

    Returns
    -------
    LineData
        The initialised LoRs.

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size`. Overlap has to be smaller than
        `sample_size`. Note that it can also be negative.

    ValueError
        If the data file does not have the (N, M >= 5) shape.


    Examples
    --------
    Initialise a `LineData` array for three LoRs on a parallel screens
    PEPT scanner (i.e. each line is defined by **two** points each) with a
    head separation of 500 mm:

    >>> lors_raw = np.array([
    >>>     [2, 100, 150, 200, 250],
    >>>     [4, 350, 250, 100, 150],
    >>>     [6, 450, 350, 250, 200]
    >>> ])

    >>> screen_separation = 500
    >>> lors = pept.scanners.parallel_screens(lors_raw, screen_separation)
    Initialised PEPT data in 0.001 s.

    >>> lors
    LineData
    --------
    sample_size = 0
    overlap =     0
    samples =     1
    lines =
      [[  2. 100. 150.   0. 200. 250. 500.]
       [  4. 350. 250.   0. 100. 150. 500.]
       [  6. 450. 350.   0. 250. 200. 500.]]
    lines.shape = (3, 7)
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']


    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    '''

    if verbose:
        start = time.time()

    # Check wheter input is a valid `pandas.read_csv` filepath or a numpy
    # array in an "Easier To Ask Forgiveness Than Permission" way.
    try:
        # Try to read the LoR data from `filepath_or_array`.
        # Check if an error is raised when reading file lines using
        lines = read_csv(filepath_or_array, **kwargs)
    except ValueError:
        # Seems like it is an array!
        lines = np.asarray(filepath_or_array, order = "C", dtype = float)

    # Add Z1 and Z2 columns => [time, X1, Y1, Z1, X2, Y2, Z2]
    # Z1 = 0
    lines = np.insert(lines, 3, 0.0, axis = 1)

    # Z2 = `separation`
    lines = np.insert(lines, 6, screen_separation, axis = 1)

    if verbose:
        end = time.time()
        print(f"Initialised PEPT data in {end - start:3.3f} s.\n")

    return LineData(lines, sample_size = sample_size, overlap = overlap)




class ADACGeometricEfficiency(PEPTObject):
    '''Compute the geometric efficiency of a parallel screens PEPT detector at
    different 3D coordinates using Antonio Guida's formula [1]_.

    The default `xlim` and `ylim` values represent the active detector area of
    the ADAC Forte scanner used at the University of Birmingham, but can be
    changed to any parallel screens detector active area range.

    This class assumes PEPT coordinates, with the Y and Z axes being swapped,
    such that Y points upwards and Z is perpendicular to the two detectors.

    Attributes
    ----------
    xlim : (2,) np.ndarray, default [111.78, 491.78]
        The limits of the active detector area in the *x*-dimension.

    ylim : (2,) np.ndarray, default [46.78, 556.78]
        The limits of the active detector area in the *y*-dimension.

    zlim : (2,) np.ndarray
        The limits of the active detector area in the *z*-dimension.

    Examples
    --------
    Simply instantiate the class with the head separation, then 'call' it with
    the (x, y, z) coordinates of the point at which to evaluate the geometric
    efficiency:

    >>> import pept
    >>> separation = 500
    >>> geom = pept.scanners.ADACGeometricEfficiency(separation)
    >>> eg = geom(250, 250, 250)

    Alternatively, the separation may be specified using the both the starting
    and ending limits:

    >>> separation = [-10, 510]
    >>> geom = pept.scanners.ADACGeometricEfficiency(separation)
    >>> eg = geom(250, 250, 250)

    You can evaluate multiple points by using a list / array of values:

    >>> geom([250, 260], 250, 250)
    array([0.18669302, 0.19730517])

    Compute the variation in geometric efficiency in the XY plane:

    >>> separation = 500
    >>> geom = pept.scanners.ADACGeometricEfficiency(separation)

    >>> # Range of x, y values to evaluate the geometric efficiency at
    >>> import numpy as np
    >>> x = np.linspace(120, 480, 100)
    >>> y = np.linspace(50, 550, 100)
    >>> z = 250

    >>> # Evaluate EG on a 2D grid of values at all combinations of x, y
    >>> xx, yy = np.meshgrid(x, y)
    >>> eg = geom(xx, yy, z)

    The geometric efficiencies can be visualised using a Plotly heatmap or
    contour plot:

    >>> import plotly.graph_objs as go
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Contour(x = x, y = y, z = eg))
    >>> fig.show()

    For an interactive 3D volumetric / voxel plot, you can use PyVista:

    >>> # Import necessary libraries; you may need to install PyVista
    >>> import numpy as np
    >>> import pept
    >>> import pyvista as pv

    >>> # Instantiate the ADACGeometricEfficiency class
    >>> geom = pept.scanners.ADACGeometricEfficiency(500)

    >>> # Lower and upper corners of the grid over which to compute the GE
    >>> lower = np.array([115, 50, 5])
    >>> upper = np.array([490, 550, 495])

    >>> # Create 3D meshgrid of values and evaluate the GE at each point
    >>> n = 40
    >>> x = np.linspace(lower[0], upper[0], n)
    >>> y = np.linspace(lower[1], upper[1], n)
    >>> z = np.linspace(lower[2], upper[2], n)
    >>> xx, yy, zz = np.meshgrid(x, y, z)
    >>> eg = geom(xx, yy, zz)

    >>> # Create PyVista grid of values
    >>> grid = pv.UniformGrid()
    >>> grid.dimensions = np.array(eg.shape) + 1
    >>> grid.origin = lower
    >>> grid.spacing = (upper - lower) / n
    >>> grid.cell_arrays["values"] = eg.flatten(order="F")

    >>> # Create PyVista volumetric / voxel plot with an interactive clipper
    >>> p = pv.Plotter()
    >>> p.add_mesh_clip_plane(grid)
    >>> p.show()

    References
    ----------
    .. [1] Guida A. Positron emission particle tracking applied to solid-liquid
       mixing in mechanically agitated vessels (Doctoral dissertation,
       University of Birmingham).

    '''

    def __init__(
        self,
        separation,
        xlim = [111.78, 491.78],
        ylim = [46.78, 556.78],
    ):
        # Separation may be either a number (e.g. 500) or a 2-list ([-10, 510])
        separation = np.array(separation)
        if separation.shape == ():
            self.zlim = np.array([0, separation])
        else:
            self.zlim = separation

        self.xlim = np.array(xlim)
        self.ylim = np.array(ylim)

        self.veg = np.vectorize(self.eg, cache = True)


    def eg(self, x, y, z):
        '''Return the geometric efficiency evaluated at a single point
        (x, y, z) *in PEPT coordinates*, i.e. Y points upwards.
        '''

        # Translate x, y, z relative to the scanner active area's centre
        x = (x - self.xlim[0]) - (self.xlim[1] - self.xlim[0]) / 2
        y = (y - self.ylim[0]) - (self.ylim[1] - self.ylim[0]) / 2
        z = (z - self.zlim[0]) - (self.zlim[1] - self.zlim[0]) / 2

        # Active detector area dimensions
        ld = self.xlim[1] - self.xlim[0]
        hd = self.ylim[1] - self.ylim[0]
        sd = self.zlim[1] - self.zlim[0]

        # Solid horizontal (XZ in PEPT coordinates) angle
        theta_min = np.arctan(max(
            (sd - 2 * z) / (ld - 2 * x),
            (sd + 2 * z) / (ld + 2 * x),
        ))

        theta_max = np.pi - np.arctan(max(
            (sd - 2 * z) / (ld + 2 * x),
            (sd + 2 * z) / (ld - 2 * x),
        ))

        # Terms for the two integrals that will be summed
        m1 = max(
            (sd - 2 * z) / (hd - 2 * y),
            (sd + 2 * z) / (hd + 2 * y),
        )

        m2 = max(
            (sd - 2 * z) / (hd + 2 * y),
            (sd + 2 * z) / (hd - 2 * y),
        )

        # Evaluate integrals
        f1_integral, _ = quad(
            lambda theta: (m1**2 / np.sin(theta)**2 + 1)**(-0.5),
            theta_min, theta_max
        )

        f2_integral, _ = quad(
            lambda theta: (m2**2 / np.sin(theta)**2 + 1)**(-0.5),
            theta_min, theta_max
        )

        return (f1_integral + f2_integral) / (2 * np.pi)


    def __call__(self, x, y, z):
        # Allow 'calling' the class using a vectorized call to `ge`
        return self.veg(x, y, z)


    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"xlim={self.xlim}, "
            f"ylim={self.ylim}, "
            f"zlim={self.zlim})"
        )
