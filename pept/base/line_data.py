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
#    Copyright (C) 2019-2021 the pept developers
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


import  pickle
from    textwrap                import  indent

import  numpy                   as      np

import  plotly.graph_objects    as      go
import  matplotlib.pyplot       as      plt

from    .iterable_samples       import  IterableSamples
from    .utilities              import  check_homogeneous_types




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

    Examples
    --------
    Initialise a `LineData` instance containing 10 lines with a `sample_size`
    of 3.

    >>> import pept
    >>> import numpy as np
    >>> lines_raw = np.arange(70).reshape(10, 7)

    >>> print(lines_raw)
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

    >>> line_data = pept.LineData(lines_raw, sample_size = 3)
    LineData
    --------
    sample_size = 3
    overlap =     0
    samples =     3
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
    lines.shape = (10, 7)
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']

    Access samples using subscript notation. Notice how the samples are
    consecutive, as `overlap` is 0 by default.

    >>> line_data[0]
    array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
           [ 7.,  8.,  9., 10., 11., 12., 13.],
           [14., 15., 16., 17., 18., 19., 20.]])

    >>> line_data[1]
    array([[21., 22., 23., 24., 25., 26., 27.],
           [28., 29., 30., 31., 32., 33., 34.],
           [35., 36., 37., 38., 39., 40., 41.]])

    Now set an overlap of 2; notice how the number of samples changes:

    >>> len(line_data)     # Number of samples
    3

    >>> line_data.overlap = 2
    >>> len(line_data)
    8

    Notice how rows are repeated from one sample to the next when accessing
    them, because `overlap` is now 2:

    >>> line_data[0]
    array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
           [ 7.,  8.,  9., 10., 11., 12., 13.],
           [14., 15., 16., 17., 18., 19., 20.]])

    >>> line_data[1]
    array([[ 7.,  8.,  9., 10., 11., 12., 13.],
           [14., 15., 16., 17., 18., 19., 20.],
           [21., 22., 23., 24., 25., 26., 27.]])

    Now change `sample_size` to 5 and notice again how the number of samples
    changes:

    >>> len(line_data)
    8

    >>> line_data.sample_size = 5
    >>> len(line_data)
    2

    >>> line_data[0]
    array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
           [ 7.,  8.,  9., 10., 11., 12., 13.],
           [14., 15., 16., 17., 18., 19., 20.],
           [21., 22., 23., 24., 25., 26., 27.],
           [28., 29., 30., 31., 32., 33., 34.]])

    >>> line_data[1]
    array([[21., 22., 23., 24., 25., 26., 27.],
           [28., 29., 30., 31., 32., 33., 34.],
           [35., 36., 37., 38., 39., 40., 41.],
           [42., 43., 44., 45., 46., 47., 48.],
           [49., 50., 51., 52., 53., 54., 55.]])

    Notice how the samples do not cover the whole input `lines_raw` array, as
    the last lines are omitted - think of the `sample_size` and `overlap`. They
    are still inside the inner `lines` attribute of `line_data` though:

    >>> line_data.lines
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

    Notes
    -----
    The class saves `lines` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the number of rows or columns after instantiating the class.

    See Also
    --------
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    pept.tracking.Cutpoints : Compute cutpoints from `pept.LineData`.
    '''

    def __init__(
        self,
        lines,
        sample_size = None,
        overlap = None,
        columns = ["t", "x1", "y1", "z1", "x2", "y2", "z2"],
        **kwargs,
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

        columns : List[str], default ["t", "x1", "y1", "z1", "x2", "y2", "z2"]
            A list of strings corresponding to the column labels in `points`.

        **kwargs : extra keyword arguments
            Any extra attributes to set on the class instance.

        Raises
        ------
        ValueError
            If `lines` has fewer than 7 columns.

        ValueError
            If `overlap` >= `sample_size` unless `sample_size` is 0. Overlap
            has to be smaller than `sample_size`. Note that it can also be
            negative.
        '''

        # Copy constructor
        if isinstance(lines, LineData):
            if sample_size is None:
                sample_size = lines.sample_size
            if overlap is None:
                overlap = lines.overlap

            # If both sample_size and overlap were None, samples_indices is
            # set differently; propagate it
            if sample_size is None and overlap is None:
                kwargs.update(samples_indices = lines.samples_indices)

            kwargs.update(lines.extra_attributes())
            kwargs.update(lines.hidden_attributes())

            lines = lines.lines

        # Create LineData from a list of LineData by stacking all inner lines
        elif len(lines) and isinstance(lines[0], LineData):
            check_homogeneous_types(lines)

            sample_size = [len(li.lines) for li in lines]
            overlap = None
            columns = lines[0].columns

            # Propagate extra attributes that were set by the user
            exclude = {"columns", "lines"}
            for k, v in lines[0].extra_attributes(exclude).items():
                setattr(self, k, v)

            lines = np.vstack([li.lines for li in lines])

        # Create LineData from a NumPy array-like
        else:
            lines = np.asarray(lines, order = 'C', dtype = float)

        # Check that lines has at least 7 columns.
        if lines.ndim != 2 or lines.shape[1] < 7:
            raise ValueError((
                "\n[ERROR]: `lines` should have dimensions (M, N), where "
                f"N >= 7. Received {lines.shape}.\n"
            ))

        self.columns = None if columns is None else [str(c) for c in columns]

        # Call the IterableSamples constructor to make the class iterable in
        # terms of samples with overlap.
        IterableSamples.__init__(self, lines, sample_size, overlap, **kwargs)


    @property
    def lines(self):
        # The `data` attribute is set by the parent class, `IterableSamples`
        return self.data


    @lines.setter
    def lines(self, lines):
        self.data = lines


    def extra_attributes(self, exclude = {}):
        exclude = set(exclude) | {"lines"}
        return IterableSamples.extra_attributes(self, exclude)


    def to_csv(self, filepath, delimiter = " "):
        '''Write `lines` to a CSV file.

        Write all LoRs stored in the class to a CSV file.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        delimiter : str, default " "
            The delimiter used to separate the values in the CSV file.

        '''

        np.savetxt(filepath, self.lines, delimiter = delimiter,
                   header = delimiter.join(self.columns))


    def save(self, filepath):
        '''Save a `LineData` instance as a binary `pickle` object.

        Saves the full object state, including the inner `.lines` NumPy array,
        `sample_size`, etc. in a fast, portable binary format. Load back the
        object using the `load` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Examples
        --------
        Save a `LineData` instance, then load it back:

        >>> lines = pept.LineData([[1, 2, 3, 4, 5, 6, 7]])
        >>> lines.save("lines.pickle")

        >>> lines_reloaded = pept.LineData.load("lines.pickle")

        '''
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(filepath):
        '''Load a saved / pickled `LineData` object from `filepath`.

        Most often the full object state was saved using the `.save` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Returns
        -------
        pept.LineData
            The loaded `pept.LineData` instance.

        Examples
        --------
        Save a `LineData` instance, then load it back:

        >>> lines = pept.LineData([[1, 2, 3, 4, 5, 6, 7]])
        >>> lines.save("lines.pickle")

        >>> lines_reloaded = pept.LineData.load("lines.pickle")

        '''
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        return obj


    def plot(
        self,
        sample_indices = ...,
        ax = None,
        alt_axes = False,
        colorbar_col = 0,
    ):
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

        ax : mpl_toolkits.mplot3D.Axes3D object, optional
            The 3D matplotlib-based axis for plotting. If undefined, new
            Matplotlib figure and axis objects are created.

        alt_axes : bool, default False
            If `True`, plot using the alternative PEPT-style axes convention:
            z is horizontal, y points upwards. Because Matplotlib cannot swap
            axes, this is achieved by swapping the parameters in the plotting
            call (i.e. `plt.plot(x, y, z)` -> `plt.plot(z, x, y)`).

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
        faster `pept.plots.PlotlyGrapher`.

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
            ax = fig.add_subplot(111, projection = '3d')
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

        if alt_axes:
            for i, line in enumerate(lines):
                ax.plot(
                    [line[3], line[6]],
                    [line[1], line[4]],
                    [line[2], line[5]],
                    c = color_array[i],
                    alpha = 0.8
                )

            ax.set_xlabel("z (mm)")
            ax.set_ylabel("x (mm)")
            ax.set_zlabel("y (mm)")

        else:
            for i, line in enumerate(lines):
                ax.plot(
                    [line[1], line[4]],
                    [line[2], line[5]],
                    [line[3], line[6]],
                    c = color_array[i],
                    alpha = 0.8
                )

            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_zlabel("z (mm)")

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
        >>> grapher = pept.plots.PlotlyGrapher()
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


    def __getitem__(self, key):
        # Allow indexing into columns
        if isinstance(key, str):
            return self.lines[:, self.columns.index(key)]

        # Otherwise use normal samples iteration
        return IterableSamples.__getitem__(self, key)


    def __repr__(self):
        # Shown when calling print(class)
        attrs = self.extra_attributes()

        with np.printoptions(threshold = 5, edgeitems = 2):
            samples_indices_str = str(self.samples_indices)
            lines_str = str(self.lines)

        return (
            "LineData\n--------\n"
            f"sample_size = {self.sample_size}\n"
            f"overlap =     {self.overlap}\n"
            f"samples =     {len(self)}\n\n"
            f"samples_indices = \n{indent(samples_indices_str, '  ')}\n\n"
            f"lines = \n{indent(lines_str, '  ')}\n\n"
            f"lines.shape = {self.lines.shape}\n\n"
        ) + "\n".join((f"{k} = {v}" for k, v in attrs.items()))
