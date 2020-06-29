#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools.
#
#    If you used this codebase or any software making use of it in a scientific
#    publication, you must cite the following paper:
#        Nicuşan AL, Windows-Yule CR. Positron emission particle tracking
#        using machine learning. Review of Scientific Instruments.
#        2020 Jan 1;91(1):013329.
#        https://doi.org/10.1063/1.5129251
#
#    Copyright (C) 2020 Andrei Leonard Nicusan
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


# File   : point_data.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 19.08.2019


import  time
import  numpy                   as      np

import  plotly.graph_objects    as      go

import  matplotlib
import  matplotlib.pyplot       as      plt
from    matplotlib.colors       import  Normalize
from    mpl_toolkits.mplot3d    import  Axes3D

from    .iterable_samples       import  IterableSamples
import  pept


class PointData(IterableSamples):
    '''A class for general PEPT point-like data iteration, manipulation and
    visualisation.

    In the context of positron-based particle tracking, points are defined by a
    timestamp, 3D coordinates and any other extra information (such as
    trajectory label or some tracer signature). This class is used for the
    encapsulation of 3D points - be they tracer locations, cutpoints, etc. -,
    efficiently yielding samples of `points` of an adaptive `sample_size` and
    `overlap`.

    Much like a complement to `LineData`, `PointData` is an abstraction over
    point-like data that may be encountered in the context of PEPT (e.g.
    pre-tracked tracer locations), as once the raw points are transformed into
    the common `PointData` format, any tracking, analysis or visualisation
    algorithm in the `pept` package can be used interchangeably. Moreover, it
    provides a stable, user-friendly interface for iterating over points in
    *samples* - this can be useful for tracking algorithms, as some take a few
    points (a *sample*), produce an accurate tracer location, then move to the
    next sample of points, repeating the procedure. Using overlapping samples
    is also useful for improving the time resolution of the algorithms.

    This is the base class for point-like data; subroutines that accept and/or
    return `PointData` instances (or subclasses thereof) can be found
    throughout the `pept` package. If you'd like to create new algorithms based
    on them, you can check out the `pept.tracking.peptml.cutpoints` module as
    an example; the `Cutpoints` class receives a `LineData` instance,
    transforms the samples of LoRs into cutpoints, then initialises itself as a
    `PointData` subclass - thereby inheriting all its methods and attributes.

    Attributes
    ----------
    points : (N, M) numpy.ndarray
        An (N, M >= 4) numpy array that stores the points as time, followed by
        cartesian (3D) coordinates of the point, followed by any extra
        information. The data columns are then `[time, x, y, z, etc]`.
    sample_size : int
        An `int` that defines the number of lines that should be returned when
        iterating over `points`. The default is 0.
    overlap : int
        An `int` that defines the overlap between two consecutive samples that
        are returned when iterating over `points`. An overlap of 0 means
        consecutive samples, while an overlap of (`sample_size` - 1) means
        incrementing the samples by one. A negative overlap means skipping
        values between samples. It is required to be smaller than
        `sample_size`. The default is 0.
    number_of_points : int
        An `int` that corresponds to len(`points`), or the number of points
        stored by `points`.
    number_of_samples : int
        An `int` that corresponds to the number of samples that can be accessed
        from the class, taking the `overlap` into consideration.

    Methods
    -------
    sample(n)
        Get sample number n (indexed from 0).
    to_csv(filepath)
        Write `points` to a CSV file.
    plot(sample_indices = ..., ax = None, colorbar_col = -1)
        Plot points from selected samples using matplotlib.
    plot_alt_axes(sample_indices = ..., ax = None, colorbar_col = -1):
        Plot points from selected samples using matplotlib on PEPT-style axes.
    points_trace(sample_indices = ..., size = 2, color = None, opacity = 0.8,\
                 colorbar = True, colorbar_col = -1, colorscale = "Magma",\
                 colorbar_title = None)
        Get a Plotly trace for all points in selected samples, with possible
        color-coding.
    copy()
        Create a deep copy of an instance of this class, including a new inner
        numpy array `points`.

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size`. Overlap is required to be smaller than
        `sample_size`, unless `sample_size` is 0. Note that it can also be
        negative.

    Notes
    -----
    This class saves `points` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the number of rows or columns after instantiating the class.

    Examples
    --------
    Initialise a `PointData` instance containing 10 points with a `sample_size`
    of 3.

    .. ipython::

        In [1]: import numpy as np

        In [2]: import pept

        In [3]: points_raw = np.arange(40).reshape(10, 4)

        In [4]: print(points_raw)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]
         [12 13 14 15]
         [16 17 18 19]
         [20 21 22 23]
         [24 25 26 27]
         [28 29 30 31]
         [32 33 34 35]
         [36 37 38 39]]

        In [5]: point_data = pept.PointData(points_raw, sample_size = 3)

        In [6]: print(point_data)
        number_of_points =  10
        sample_size =       3
        overlap =           0
        number_of_samples = 3
        points =
        [[ 0.  1.  2.  3.]
         [ 4.  5.  6.  7.]
         [ 8.  9. 10. 11.]
         [12. 13. 14. 15.]
         [16. 17. 18. 19.]
         [20. 21. 22. 23.]
         [24. 25. 26. 27.]
         [28. 29. 30. 31.]
         [32. 33. 34. 35.]
         [36. 37. 38. 39.]]

    Access samples using subscript notation. Notice how the samples are
    consecutive, as `overlap` is 0 by default.

    .. ipython::

        In [7]: point_data[0]
        Out[7]:
        array([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.]])

        In [8]: point_data[1]
        Out[8]:
        array([[12., 13., 14., 15.],
               [16., 17., 18., 19.],
               [20., 21., 22., 23.]])

    Now set an overlap of 2; notice how the number of samples changes:

    .. ipython::

        In [9]: len(point_data)         # Number of samples
        Out[9]: 3

        In [10]: point_data.overlap = 2

        In [11]: len(point_data)
        Out[11]: 8

    Notice how rows are repeated from one sample to the next when accessing
    them, because `overlap` is now 2:

    .. ipython::

        In [12]: point_data[0]
        Out[12]:
        array([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.]])

        In [13]: point_data[1]
        Out[13]:
        array([[ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.],
               [12., 13., 14., 15.]])

    Now change `sample_size` to 5 and notice again how the number of samples
    changes:

    .. ipython::

        In [14]: len(point_data)
        Out[14]: 8

        In [15]: point_data.sample_size = 5

        In [16]: len(point_data)
        Out[16]: 2

        In [17]: point_data[0]
        Out[17]:
        array([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.],
               [12., 13., 14., 15.],
               [16., 17., 18., 19.]])

        In [18]: point_data[1]
        Out[18]:
        array([[12., 13., 14., 15.],
               [16., 17., 18., 19.],
               [20., 21., 22., 23.],
               [24., 25., 26., 27.],
               [28., 29., 30., 31.]])

    Notice how the samples do not cover the whole input `points_raw` array, as
    the last lines are omitted - think of the `sample_size` and `overlap`. They
    are still inside the inner `points` attribute of `point_data` though:

    .. ipython::

        In [19]: point_data.points
        Out[19]:
        array([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.],
               [12., 13., 14., 15.],
               [16., 17., 18., 19.],
               [20., 21., 22., 23.],
               [24., 25., 26., 27.],
               [28., 29., 30., 31.],
               [32., 33., 34., 35.],
               [36., 37., 38., 39.]])

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    pept.tracking.peptml.Cutpoints : Compute cutpoints from `pept.LineData`.
    '''

    def __init__(
        self,
        points,
        sample_size = 0,
        overlap = 0,
        verbose = False
    ):
        '''`PointData` class constructor.

        Parameters
        ----------
        points : (N, M) numpy.ndarray
            An (N, M >= 4) numpy array that stores points (or any generic 2D
            set of data). It expects that the first column is time, followed by
            cartesian (3D) coordinates of points, followed by any extra
            information the user needs. The data columns are then
            `[time, x, y, z, etc]`.
        sample_size : int, default 0
            An `int`` that defines the number of points that should be returned
            when iterating over `points`. A `sample_size` of 0 yields all the
            data as one single sample.
        overlap : int, default 0
            An `int` that defines the overlap between two consecutive samples
            that are returned when iterating over `points`. An overlap of 0
            means consecutive samples, while an overlap of (`sample_size` - 1)
            implies incrementing the samples by one. A negative overlap means
            skipping values between samples. An error is raised if `overlap` is
            larger than or equal to `sample_size`.
        verbose : bool, default False
            An option that enables printing the time taken for the
            initialisation of an instance of the class. Useful when reading
            large files (10gb files for PEPT data is not unheard of).

        Raises
        ------
        ValueError
            If `line_data` does not have (N, M) shape, where M >= 4.
        '''

        if verbose:
            start = time.time()

        # If `points` is not C-contiguous, create a C-contiguous copy.
        self._points = np.asarray(points, order = 'C', dtype = float)

        # Check that points has at least 4 columns.
        if self._points.ndim != 2 or self._points.shape[1] < 4:
            raise ValueError((
                "\n[ERROR]: `points` should have dimensions (M, N), where "
                f"N >= 4. Received {self._points.shape}.\n"
            ))

        self._number_of_points = len(self._points)

        # Call the IterableSamples constructor to make the class iterable in
        # samples with overlap.
        IterableSamples.__init__(self, sample_size, overlap)

        if verbose:
            end = time.time()
            print(f"Initialising the PEPT data took {end - start} seconds.\n")


    @property
    def points(self):
        '''Get the points stored in the class.
        '''

        return self._points


    @property
    def data_samples(self):
        '''Implemented property for the `IterableSamples` parent class. See its
        documentation for more information.
        '''

        return self._points


    @property
    def data_length(self):
        '''Implemented property for the IterableSamples parent class. See its
        documentation for more information.
        '''

        return self._number_of_points


    @property
    def number_of_points(self):
        '''Get the number of points stored in the class.
        '''

        return self._number_of_points


    def to_csv(self, filepath):
        '''Write `points` to a CSV file.

        Write all points (and any extra data) stored in the class to a CSV
        file.

        Parameters
        ----------
            filepath : filename or file handle
                If filepath is a path (rather than file handle), it is relative
                to where python is called.
        '''

        np.savetxt(filepath, self._points)


    def plot(self, sample_indices = ..., ax = None, colorbar_col = -1):
        '''Plot points from selected samples using matplotlib.

        Returns matplotlib figure and axes objects containing all points
        included in the samples selected by `sample_indices`.
        `sample_indices` may be a single sample index (e.g. 0), an iterable
        of indices (e.g. [1,5,6]), or an Ellipsis (`...`) for all samples.

        Parameters
        ----------
        sample_indices : int or iterable or Ellipsis, default Ellipsis
            The index or indices of the samples of points. An `int` signifies
            the sample index, an iterable (list-like) signifies multiple sample
            indices, while an Ellipsis (`...`) signifies all samples. The
            default is `...` (all points).
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.
        colorbar_col : int, default -1
            The column in the data samples that will be used to color the
            points. The default is -1 (the last column).

        Returns
        -------
        fig, ax : matplotlib figure and axes objects

        Notes
        -----
        Plotting all points is very computationally-expensive for matplotlib.
        It is recommended to only plot a couple of samples at a time, or use
        the faster `pept.visualisation.PlotlyGrapher`.

        Examples
        --------
        Plot the points from sample 1 in a `PointData` instance:

        >>> point_data = pept.PointData(...)
        >>> fig, ax = point_data.plot(1)
        >>> fig.show()

        Plot the points from samples 0, 1 and 2:

        >>> fig, ax = point_data.plot([0, 1, 2])
        >>> fig.show()

        '''

        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()

        # Check if sample_indices is an iterable collection (list-like)
        # otherwise just "iterate" over the single number or Ellipsis
        if not hasattr(sample_indices, "__iter__"):
            sample_indices = [sample_indices]

        if sample_indices[0] == Ellipsis:
            x = self._points[:, 1],
            y = self._points[:, 2],
            z = self._points[:, 3],
            color = self._points[:, colorbar_col]
        else:
            x, y, z, color = [], [], [], []
            for n in sample_indices:
                sample = self[n]

                x.extend(sample[:, 1])
                y.extend(sample[:, 2])
                z.extend(sample[:, 3])

                color.extend(sample[:, colorbar_col])

        color = np.array(color)

        # Scatter x, y, z, [color]
        cmap = plt.cm.magma
        color_array = cmap(color / color.max())

        ax.scatter(x, y, z, c = color_array)

        return fig, ax


    def plot_alt_axes(self, sample_indices = ..., ax = None,
                      colorbar_col = -1):
        '''Plot points from selected samples using matplotlib on PEPT-style
        axes.

        Returns matplotlib figure and axes objects containing all points
        included in the samples selected by `sample_indices`.
        `sample_indices` may be a single sample index (e.g. 0), an iterable
        of indices (e.g. [1,5,6]), or an Ellipsis (`...`) for all samples.

        The points are plotted using the PEPT-style convention: **x** is
        *parallel and horizontal* to the screens, **y** is
        *parallel and vertical* to the screens, **z** is *perpendicular* to the
        screens. The mapping relative to the Cartesian coordinates would then
        be: (x, y, z) -> (z, x, y).

        Parameters
        ----------
        sample_indices : int or iterable or Ellipsis, default Ellipsis
            The index or indices of the samples of points. An `int` signifies
            the sample index, an iterable (list-like) signifies multiple sample
            indices, while an Ellipsis (`...`) signifies all samples. The
            default is `...` (all points).
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.
        colorbar_col : int, default -1
            The column in the data samples that will be used to color the
            points. The default is -1 (the last column).

        Returns
        -------
        fig, ax : matplotlib figure and axes objects

        Notes
        -----
        Plotting all points is very computationally-expensive for matplotlib.
        It is recommended to only plot a couple of samples at a time, or use
        the faster `pept.visualisation.PlotlyGrapher`.

        Examples
        --------
        Plot the points from sample 1 in a `PointData` instance:

        >>> point_data = pept.PointData(...)
        >>> fig, ax = point_data.plot_alt_axes(1)
        >>> fig.show()

        Plot the points from samples 0, 1 and 2:

        >>> fig, ax = point_data.plot_alt_axes([0, 1, 2])
        >>> fig.show()

        '''

        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()

        # Check if sample_indices is an iterable collection (list-like)
        # otherwise just "iterate" over the single number or Ellipsis
        if not hasattr(sample_indices, "__iter__"):
            sample_indices = [sample_indices]

        if sample_indices[0] == Ellipsis:
            x = self._points[:, 1],
            y = self._points[:, 2],
            z = self._points[:, 3],
            color = self._points[:, colorbar_col]
        else:
            x, y, z, color = [], [], [], []
            for n in sample_indices:
                sample = self[n]

                x.extend(sample[:, 1])
                y.extend(sample[:, 2])
                z.extend(sample[:, 3])

                color.extend(sample[:, colorbar_col])

        color = np.array(color)

        # Scatter x, y, z, [color]
        cmap = plt.cm.magma
        color_array = cmap(color / color.max())

        ax.scatter(z, x, y, c = color_array)

        return fig, ax


    def points_trace(
        self,
        sample_indices = ...,
        size = 2,
        color = None,
        opacity = 0.8,
        colorbar = True,
        colorbar_col = -1,
        colorscale = "Magma",
        colorbar_title = None
    ):
        '''Get a Plotly trace for all points in selected samples, with possible
        color-coding.

        Returns a `plotly.graph_objects.Scatter3d` trace containing all points
        included in in the samples selected by `sample_indices`.
        `sample_indices` may be a single sample index (e.g. 0), an iterable
        of indices (e.g. [1,5,6]), or an Ellipsis (`...`) for all samples.

        Can then be passed to the `plotly.graph_objects.figure.add_trace`
        function or a `PlotlyGrapher` instance using the `add_trace` method.

        Parameters
        ----------
        sample_indices : int or iterable or Ellipsis, default Ellipsis
            The index or indices of the samples of points. An `int` signifies
            the sample index, an iterable (list-like) signifies multiple sample
            indices, while an Ellipsis (`...`) signifies all samples. The
            default is `...` (all points).
        size : float, default 2
            The marker size of the points.
        color : str or list-like, optional
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Overrides `colorbar` if set. For more information,
            check the Plotly documentation.
        opacity : float, default 0.8
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque.
        colorbar : bool, default True
            If set to True, will color-code the data in the sample column
            `colorbar_col`. Is overridden if `color` is set.
        colorbar_col : int, default -1
            The column in the data samples that will be used to color the
            points. Only has an effect if `colorbar` is set to True. The
            default is -1 (the last column).
        colorscale : str, default "Magma"
            The Plotly scheme for color-coding the `colorbar_col` column in the
            input data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.
        colorbar_title : str, optional
            If set, the colorbar will have this title above.

        Returns
        -------
        plotly.graph_objs.Scatter3d
            A Plotly trace of the points.

        Examples
        --------
        Use `PlotlyGrapher` (a user-friendly wrapper around the `plotly`
        library for PEPT-oriented data) to plot the points from sample 1 in a
        `PointData` instance:

        >>> point_data = pept.PointData(...)
        >>> grapher = pept.visualisation.PlotlyGrapher()
        >>> trace = point_data.points_trace(1)
        >>> grapher.add_trace(trace)
        >>> grapher.show()

        Use `plotly.graph_objs` to plot the lines from samples 0, 1 and 2:

        >>> import plotly.graph_objs as go
        >>> fig = go.Figure()
        >>> fig.add_trace(point_data.points_trace([0, 1, 2]))
        >>> fig.show()

        '''

        # Check if sample_indices is an iterable collection (list-like)
        # otherwise just "iterate" over the single number or Ellipsis
        if not hasattr(sample_indices, "__iter__"):
            sample_indices = [sample_indices]

        coords_x = []
        coords_y = []
        coords_z = []

        marker = dict(
            size = size,
            color = color,
            opacity = opacity
        )

        if colorbar:
            if color is None:
                marker['color'] = []

            marker.update(colorscale = colorscale)
            if colorbar_title is not None:
                marker.update(colorbar = dict(title = colorbar_title))

        # If an Ellipsis was received, include all points
        if sample_indices[0] is Ellipsis:
            coords_x = self.points[:, 1]
            coords_y = self.points[:, 2]
            coords_z = self.points[:, 3]

            if colorbar and color is None:
                marker['color'] = self.points[:, colorbar_col]
        else:
            # For each selected sample include all the needed coordinates
            for n in sample_indices:
                sample = self[n]

                coords_x.extend(sample[:, 1])
                coords_y.extend(sample[:, 2])
                coords_z.extend(sample[:, 3])

                if colorbar and color is None:
                    marker['color'].extend(sample[:, colorbar_col])

        trace = go.Scatter3d(
            x = coords_x,
            y = coords_y,
            z = coords_z,
            mode = "markers",
            marker = marker
        )

        return trace


    def copy(self):
        '''Create a deep copy of an instance of this class, including a new
        inner numpy array `points`.

        Returns
        -------
        pept.PointData
            A new instance of the `pept.PointData` class with the same
            attributes as this instance, deep-copied.
        '''

        return pept.PointData(
            self._points.copy(order = "C"),
            sample_size = self._sample_size,
            overlap = self._overlap,
            verbose = False
        )


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"number_of_points =  {self.number_of_points}\n\n"
            f"sample_size =       {self._sample_size}\n"
            f"overlap =           {self._overlap}\n"
            f"number_of_samples = {self.number_of_samples}\n\n"
            f"points = \n{self._points}"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPL

        docstr = (
            "Class instance that inherits from `pept.PointData`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n----------\n"
            f"{self.__str__()}\n\n"
            "Particular Cases\n----------------\n"
            " > If sample_size == 0, all `points` are returned as a "
               "single sample.\n"
            " > If overlap >= sample_size, an error is raised.\n"
            " > If overlap < 0, points are skipped between samples."
        )

        return docstr


