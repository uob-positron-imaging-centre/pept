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


# File   : line_data.py
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


class LineData(IterableSamples):
    '''A class for PEPT LoR data iteration, manipulation and visualisation.

    Generally, PEPT Lines of Response (LoRs) are lines in 3D space, each
    defined by two points, regardless of the geometry of the scanner used. This
    class is used for the encapsulation of LoRs (or any lines!), efficiently
    yielding samples of `lines` of an adaptive `sample_size` and `overlap`.

    It is an abstraction over PET / PEPT scanner geometries and data formats,
    as once the raw LoRs (be they stored as binary, ASCII, etc.) are
    transformed into the common `LineData` format, any tracking, analysis or
    visualisation algorithm in the `pept` package can be used interchangeably.
    Moreover, it provides a stable, user-friendly interface for iterating over
    LoRs in *samples* - this is useful for tracking algorithms, as they
    generally take a few LoRs (a *sample*), produce a tracer position, then
    move to the next sample of LoRs, repeating the procedure. Using overlapping
    samples is also useful for improving the time resolution of the algorithms.

    This is the base class for LoR data; the subroutines for transforming other
    data formats into `LineData` can be found in `pept.scanners`. If you'd like
    to integrate another scanner geometry or raw data format into this package,
    you can check out the `pept.scanners.parallel_screens` module as an
    example. This usually only involves writing a single function by hand; then
    all attributes and methods from `LineData` will be available to your new
    data format. If you'd like to use `LineData` as the base for other
    algorithms, you can check out the `pept.tracking.peptml.cutpoints` module
    as an example; the `Cutpoints` class iterates the samples of LoRs in any
    `LineData` **in parallel**, using `concurrent.futures.ThreadPoolExecutor`.

    Attributes
    ----------
    lines : (N, M>=7) numpy.ndarray
        An (N, M>=7) numpy array that stores the PEPT LoRs as time and
        cartesian (3D) coordinates of two points defining a line, followed by
        any additional data. The data columns are then
        `[time, x1, y1, z1, x2, y2, z2, etc.]`.
    sample_size : int
        An `int` that defines the number of lines that should be returned when
        iterating over `lines`. The default is 0.
    overlap : int
        An `int` that defines the overlap between two consecutive samples that
        are returned when iterating over `lines`. An overlap of 0 implies
        consecutive samples, while an overlap of (`sample_size` - 1) implies
        incrementing the samples by one. A negative overlap means skipping
        values between samples. It is required to be smaller than
        `sample_size`. The default is 0.
    number_of_lines : int
        An `int` that corresponds to len(`lines`), or the number of LoRs
        stored by `lines`.
    number_of_samples : int
        An `int` that corresponds to the number of samples that can be accessed
        from the class. It takes `overlap` into consideration.

    Methods
    -------
    sample(n)
        Get sample number n (indexed from 0).
    to_csv(filepath)
        Write `lines` to a CSV file.
    plot(sample_indices = ..., ax = None, colorbar_col = 0)
        Plot lines from selected samples using matplotlib.
    plot_alt_axes(sample_indices = ..., ax = None, colorbar_col = 0):
        Plot lines from selected samples using matplotlib on PEPT-style axes.
    lines_trace(sample_indices = ..., width = 2, color = None, opacity = 0.6,\
                colorbar = True, colorbar_col = 0, colorbar_title = None)
        Get a Plotly trace for all the lines in selected samples.
    copy()
        Create a deep copy of an instance of this class, including a new inner
        numpy array `lines`.

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size` unless `sample_size` is 0. Overlap
        has to be smaller than `sample_size`. Note that it can also be
        negative.

    Notes
    -----
    The class saves `lines` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the number of rows or columns after instantiating the class.

    Examples
    --------
    Initialise a `LineData` instance containing 10 lines with a `sample_size`
    of 3.

    .. ipython::

        In [3]: import pept

        In [4]: import numpy as np

        In [5]: lines_raw = np.arange(70).reshape(10, 7)

        In [6]: print(lines_raw)
        [[ 0  1  2  3  4  5  6]
         [ 7  8  9 10 11 12 13]
         [14 15 16 17 18 19 20]
         [21 22 23 24 25 26 27]
         [28 29 30 31 32 33 34]
         [35 36 37 38 39 40 41]
         [42 43 44 45 46 47 48]
         [49 50 51 52 53 54 55]
         [56 57 58 59 60 61 62]
         [63 64 65 66 67 68 69]]

        In [7]: line_data = pept.LineData(lines_raw, sample_size = 3)

        In [8]: print(line_data)
        number_of_lines =   10
        sample_size =       3
        overlap =           0
        number_of_samples = 3
        lines =
        [[ 0.  1.  2.  3.  4.  5.  6.]
         [ 7.  8.  9. 10. 11. 12. 13.]
         [14. 15. 16. 17. 18. 19. 20.]
         [21. 22. 23. 24. 25. 26. 27.]
         [28. 29. 30. 31. 32. 33. 34.]
         [35. 36. 37. 38. 39. 40. 41.]
         [42. 43. 44. 45. 46. 47. 48.]
         [49. 50. 51. 52. 53. 54. 55.]
         [56. 57. 58. 59. 60. 61. 62.]
         [63. 64. 65. 66. 67. 68. 69.]]

    Access samples using subscript notation. Notice how the samples are
    consecutive, as `overlap` is 0 by default.

    .. ipython::

        In [9]: line_data[0]
        Out[9]:
        array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
               [ 7.,  8.,  9., 10., 11., 12., 13.],
               [14., 15., 16., 17., 18., 19., 20.]])

        In [10]: line_data[1]
        Out[10]:
        array([[21., 22., 23., 24., 25., 26., 27.],
               [28., 29., 30., 31., 32., 33., 34.],
               [35., 36., 37., 38., 39., 40., 41.]])

    Now set an overlap of 2; notice how the number of samples changes:

    .. ipython::

        In [11]: len(line_data)     # Number of samples
        Out[11]: 3

        In [12]: line_data.overlap = 2

        In [13]: len(line_data)
        Out[13]: 8

    Notice how rows are repeated from one sample to the next when accessing
    them, because `overlap` is now 2:

    .. ipython::

        In [14]: line_data[0]
        Out[14]:
        array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
               [ 7.,  8.,  9., 10., 11., 12., 13.],
               [14., 15., 16., 17., 18., 19., 20.]])

        In [15]: line_data[1]
        Out[15]:
        array([[ 7.,  8.,  9., 10., 11., 12., 13.],
               [14., 15., 16., 17., 18., 19., 20.],
               [21., 22., 23., 24., 25., 26., 27.]])

    Now change `sample_size` to 5 and notice again how the number of samples
    changes:

    .. ipython::

        In [16]: len(line_data)
        Out[16]: 8

        In [17]: line_data.sample_size = 5

        In [18]: len(line_data)
        Out[18]: 2

        In [19]: line_data[0]
        Out[19]:
        array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
               [ 7.,  8.,  9., 10., 11., 12., 13.],
               [14., 15., 16., 17., 18., 19., 20.],
               [21., 22., 23., 24., 25., 26., 27.],
               [28., 29., 30., 31., 32., 33., 34.]])

        In [20]: line_data[1]
        Out[20]:
        array([[21., 22., 23., 24., 25., 26., 27.],
               [28., 29., 30., 31., 32., 33., 34.],
               [35., 36., 37., 38., 39., 40., 41.],
               [42., 43., 44., 45., 46., 47., 48.],
               [49., 50., 51., 52., 53., 54., 55.]])

    Notice how the samples do not cover the whole input `lines_raw` array, as
    the last lines are omitted - think of the `sample_size` and `overlap`. They
    are still inside the inner `lines` attribute of `line_data` though:

    .. ipython::

        In [21]: line_data.lines
        Out[21]:
        array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
               [ 7.,  8.,  9., 10., 11., 12., 13.],
               [14., 15., 16., 17., 18., 19., 20.],
               [21., 22., 23., 24., 25., 26., 27.],
               [28., 29., 30., 31., 32., 33., 34.],
               [35., 36., 37., 38., 39., 40., 41.],
               [42., 43., 44., 45., 46., 47., 48.],
               [49., 50., 51., 52., 53., 54., 55.],
               [56., 57., 58., 59., 60., 61., 62.],
               [63., 64., 65., 66., 67., 68., 69.]])

    See Also
    --------
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    pept.tracking.peptml.Cutpoints : Compute cutpoints from `pept.LineData`.
    '''

    def __init__(
        self,
        lines,
        sample_size = 0,
        overlap = 0,
        verbose = False
    ):
        '''`LineData` class constructor.

        Parameters
        ----------
        lines : (N, M>=7) numpy.ndarray
            An (N, M>=7) numpy array that stores the PEPT LoRs (or any generic
            set of lines) as time and cartesian (3D) coordinates of two points
            defining each line, followed by any additional data. The data
            columns are then `[time, x1, y1, z1, x2, y2, z2, etc.]`.
        sample_size : int, default 0
            An `int` that defines the number of lines that should be returned
            when iterating over `lines`. A `sample_size` of 0 yields all the
            data as one single sample.
        overlap : int, default 0
            An `int` that defines the overlap between two consecutive samples
            that are returned when iterating over `lines`. An overlap of 0
            means consecutive samples, while an overlap of (`sample_size` - 1)
            means incrementing the samples by one. A negative overlap means
            skipping values between samples. An error is raised if `overlap` is
            larger than or equal to `sample_size`.
        verbose : bool, default False
            An option that enables printing the time taken for the
            initialisation of an instance of the class. Useful when reading
            large files (10gb files for PEPT data is not unheard of).

        Raises
        ------
        ValueError
            If `lines` has fewer than 7 columns.
        ValueError
            If `overlap` >= `sample_size` unless `sample_size` is 0. Overlap
            has to be smaller than `sample_size`. Note that it can also be
            negative.
        '''

        if verbose:
            start = time.time()

        # If `lines` is not C-contiguous, create a C-contiguous copy
        self._lines = np.asarray(lines, order = 'C', dtype = float)

        # Check that lines has at least 7 columns.
        if self._lines.ndim != 2 or self._lines.shape[1] < 7:
            raise ValueError((
                "\n[ERROR]: `lines` should have dimensions (M, N), where "
                f"N >= 7. Received {self._lines.shape}.\n"
            ))

        self._number_of_lines = len(self._lines)

        # Call the IterableSamples constructor to make the class iterable in
        # terms of samples with overlap.
        IterableSamples.__init__(self, sample_size, overlap)

        if verbose:
            end = time.time()
            print(f"Initialising the line data took {end - start} seconds.\n")


    @property
    def lines(self):
        '''The lines stored in the class.
        '''

        return self._lines


    @property
    def data_samples(self):
        '''Implemented property for the IterableSamples parent class. See its
        documentation for more information.
        '''

        return self._lines


    @property
    def data_length(self):
        '''Implemented property for the IterableSamples parent class. See its
        documentation for more information.
        '''

        return self._number_of_lines


    @property
    def number_of_lines(self):
        '''The number of lines stored in the class.
        '''

        return self._number_of_lines


    def to_csv(self, filepath):
        '''Write `lines` to a CSV file.

        Write all LoRs stored in the class to a CSV file.

        Parameters
        ----------
            filepath : filename or file handle
                If filepath is a path (rather than file handle), it is relative
                to where python is called.
        '''

        np.savetxt(filepath, self._lines, delimiter = delimiter)


    def plot(self, sample_indices = ..., ax = None, colorbar_col = 0):
        '''Plot lines from selected samples using matplotlib.

        Returns matplotlib figure and axes objects containing all lines
        included in the samples selected by `sample_indices`.
        `sample_indices` may be a single sample index (e.g. 0), an iterable
        of indices (e.g. [1,5,6]), or an Ellipsis (`...`) for all samples.

        Parameters
        ----------
        sample_indices : int or iterable or Ellipsis, default Ellipsis
            The index or indices of the samples of lines. An `int` signifies
            the sample index, an iterable (list-like) signifies multiple sample
            indices, while an Ellipsis (`...`) signifies all samples. The
            default is `...` (all lines).
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.
        colorbar_col : int, default -1
            The column in the data samples that will be used to color the
            lines. The default is -1 (the last column).

        Returns
        -------
        fig, ax : matplotlib figure and axes objects

        Notes
        -----
        Plotting all lines is very computationally-expensive for matplotlib. It
        is recommended to only plot a couple of samples at a time, or use the
        faster `pept.visualisation.PlotlyGrapher`.

        Examples
        --------
        Plot the lines from sample 1 in a `LineData` instance:

        >>> lors = pept.LineData(...)
        >>> fig, ax = lors.plot(1)
        >>> fig.show()

        Plot the lines from samples 0, 1 and 2:

        >>> fig, ax = lors.plot([0, 1, 2])
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

        lines, color = [], []
        # For each selected sample include all the lines' coordinates
        for n in sample_indices:
            # If an Ellipsis was received, then include all lines.
            if n is Ellipsis:
                sample = self.lines
            else:
                sample = self[n]

            for line in sample:
                lines.append(line[0:7])
                color.append(line[colorbar_col])

        color = np.array(color)

        # Scatter x, y, z [color]
        cmap = plt.cm.magma
        color_array = cmap(color / color.max())

        for i, line in enumerate(lines):
            ax.plot(
                [line[1], line[4]],
                [line[2], line[5]],
                [line[3], line[6]],
                c = color_array[i],
                alpha = 0.8
            )

        return fig, ax


    def plot_alt_axes(self, sample_indices = ..., ax = None, colorbar_col = 0):
        '''Plot lines from selected samples using matplotlib on PEPT-style
        axes.

        Returns matplotlib figure and axes objects containing all lines
        included in the samples selected by `sample_indices`.
        `sample_indices` may be a single sample index (e.g. 0), an iterable
        of indices (e.g. [1,5,6]), or an Ellipsis (`...`) for all samples.

        The lines are plotted using the PEPT-style convention: **x** is
        *parallel and horizontal* to the screens, **y** is
        *parallel and vertical* to the screens, **z** is *perpendicular* to the
        screens. The mapping relative to the Cartesian coordinates would then
        be: (x, y, z) -> (z, x, y).

        Parameters
        ----------
        sample_indices : int or iterable or Ellipsis, default Ellipsis
            The index or indices of the samples of lines. An `int` signifies
            the sample index, an iterable (list-like) signifies multiple sample
            indices, while an Ellipsis (`...`) signifies all samples. The
            default is `...` (all lines).
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.
        colorbar_col : int, default -1
            The column in the data samples that will be used to color the
            lines. The default is -1 (the last column).

        Returns
        -------
        fig, ax : matplotlib figure and axes objects

        Notes
        -----
        Plotting all lines is very computationally-expensive for matplotlib. It
        is recommended to only plot a couple of samples at a time, or use the
        faster `pept.visualisation.PlotlyGrapher`.

        Examples
        --------
        Plot the lines from sample 1 in a `LineData` instance:

        >>> lors = pept.LineData(...)
        >>> fig, ax = lors.plot_alt_axes(1)
        >>> fig.show()

        Plot the lines from samples 0, 1 and 2:

        >>> fig, ax = lors.plot_alt_axes([0, 1, 2])
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

        lines, color = [], []
        # For each selected sample include all the lines' coordinates
        for n in sample_indices:
            # If an Ellipsis was received, then include all lines.
            if n is Ellipsis:
                sample = self.lines
            else:
                sample = self[n]

            for line in sample:
                lines.append(line[0:7])
                color.append(line[colorbar_col])

        color = np.array(color)

        # Scatter x, y, z [color]
        cmap = plt.cm.magma
        color_array = cmap(color / color.max())

        for i, line in enumerate(lines):
            ax.plot(
                [line[3], line[6]],
                [line[1], line[4]],
                [line[2], line[5]],
                c = color_array[i],
                alpha = 0.8
            )

        return fig, ax


    def lines_trace(
        self,
        sample_indices = ...,
        width = 2.0,
        color = None,
        opacity = 0.6,
        colorbar = True,
        colorbar_col = 0,
        colorscale = "Magma",
        colorbar_title = None
    ):
        '''Get a Plotly trace for all the lines in selected samples.

        Creates a `plotly.graph_objects.Scatter3d` object for all the lines
        included in the samples selected by `sample_indices`. `sample_indices`
        may be a single sample index (e.g. 0), an iterable of indices (e.g.
        [1,5,6]) or an Ellipsis (`...`) for all samples.

        Can then be passed to the `plotly.graph_objects.figure.add_trace`
        function or a `PlotlyGrapher` instance using the `add_trace` method.

        Parameters
        ----------
        sample_indices : int or iterable or Ellipsis, default Ellipsis
            The index or indices of the samples of LoRs. An `int` signifies the
            sample index, an iterable (list-like) signifies multiple sample
            indices, while an Ellipsis (`...`) signifies all samples. The
            default is `...` (all lines).
        width : float, default 2.0
            The width of the lines.
        color : str or list-like, optional
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Overrides `colorbar` if set. For more information,
            check the Plotly documentation. The default is None.
        opacity : float, default 0.6
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque.
        colorbar : bool, default True
            If set to True, will color-code the data in the sample column
            `colorbar_col`. Is overridden if `color` is set. The default is
            True, so that every line has a different color.
        colorbar_col : int, default 0
            The column in the data samples that will be used to color the
            points. Only has an effect if `colorbar` is set to True. The
            default is 0 (the first column - time).
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
            A Plotly trace of the LoRs.

        Examples
        --------
        Use `PlotlyGrapher` (a user-friendly wrapper around the `plotly`
        library for PEPT-oriented data) to plot the lines from sample 1 in a
        `LineData` instance:

        >>> lors = pept.LineData(...)
        >>> grapher = pept.visualisation.PlotlyGrapher()
        >>> trace = lors.lines_trace(1)
        >>> grapher.add_trace(trace)
        >>> grapher.show()

        Use `plotly.graph_objs` to plot the lines from samples 0, 1 and 2:

        >>> import plotly.graph_objs as go
        >>> fig = go.Figure()
        >>> fig.add_trace(lors.lines_trace([0, 1, 2]))
        >>> fig.show()

        '''

        # Check if sample_indices is an iterable collection (list-like)
        # otherwise just "iterate" over the single number or Ellipsis.
        if not hasattr(sample_indices, "__iter__"):
            sample_indices = [sample_indices]

        marker = dict(
            width = width,
            color = color,
        )

        if colorbar:
            if color is None:
                marker['color'] = []

            marker.update(colorscale = colorscale)
            if colorbar_title is not None:
                marker.update(colorbar = dict(title = colorbar_title))

        coords_x = []
        coords_y = []
        coords_z = []

        # For each selected sample include all the lines' coordinates
        for n in sample_indices:
            # If an Ellipsis was received, then include all lines.
            if n is Ellipsis:
                sample = self.lines
            else:
                sample = self[n]

            for line in sample:
                coords_x.extend([line[1], line[4], None])
                coords_y.extend([line[2], line[5], None])
                coords_z.extend([line[3], line[6], None])

                if colorbar and color is None:
                    marker['color'].extend(3 * [line[colorbar_col]])

        trace = go.Scatter3d(
            x = coords_x,
            y = coords_y,
            z = coords_z,
            mode = 'lines',
            opacity = opacity,
            line = marker
        )

        return trace


    def copy(self):
        '''Create a deep copy of an instance of this class, including a new
        inner numpy array `lines`.

        Returns
        -------
        pept.LineData
            A new instance of the `pept.LineData` class with the same
            attributes as this instance, deep-copied.
        '''

        return pept.LineData(
            self._lines.copy(order = "C"),
            sample_size = self._sample_size,
            overlap = self._overlap,
            verbose = False
        )


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"number_of_lines =   {self.number_of_lines}\n\n"
            f"sample_size =       {self._sample_size}\n"
            f"overlap =           {self._overlap}\n"
            f"number_of_samples = {self.number_of_samples}\n\n"
            f"lines = \n{self._lines}"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPR
        docstr = (
            "Class instance that inherits from `pept.LineData`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n----------\n"
            f"{self.__str__()}\n\n"
            "Particular Cases\n----------------\n"
            " > If sample_size == 0, all `lines` are returned as a "
               "single sample.\n"
            " > If overlap >= sample_size, an error is raised.\n"
            " > If overlap < 0, lines are skipped between samples."
        )

        return docstr


