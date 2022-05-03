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


from    textwrap                import  indent

import  numpy                   as      np

import  matplotlib.pyplot       as      plt

from    .iterable_samples       import  IterableSamples




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
    samples is also useful for improving the tracking rate of the algorithms.

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

    sample_size : int, list[int], pept.TimeWindow or None
        Defining the number of LoRs in a sample; if it is an integer, a
        constant number of LoRs are returned per sample. If it is a list of
        integers, sample `i` will have length `sample_size[i]`. If it is a
        `pept.TimeWindow` instance, each sample will span a fixed time window.
        If `None`, custom sample sizes are returned as per the
        `samples_indices` attribute.

    overlap : int, pept.TimeWindow or None
        Defining the overlapping LoRs between consecutive samples. If `int`,
        constant numbers of LoRs are used. If `pept.TimeWindow`, the overlap
        will be a constant time window across the data timestamps (first
        column). If `None`, custom sample sizes are defined as per the
        `samples_indices` attribute.

    samples_indices : (S, 2) numpy.ndarray
        A 2D NumPy array of integers, where row `i` defines the i-th sample's
        start and end row indices, i.e.
        `sample[i] == data[samples_indices[i, 0]:samples_indices[i, 1]]`. The
        `sample_size` and `overlap` are simply friendly interfaces to setting
        the `samples_indices`.

    columns : (M,) list[str]
        A list of strings with the same number of columns as `lines` containing
        each column's name.

    attrs : dict[str, Any]
        A dictionary of other attributes saved on this class. Attribute names
        starting with an underscore are considered "hidden".

    See Also
    --------
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    pept.tracking.Cutpoints : Compute cutpoints from `pept.LineData`.

    Notes
    -----
    The class saves `lines` as a **C-contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the number of rows or columns after instantiating the class.

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
    >>> line_data
    pept.LineData (samples: 3)
    --------------------------
    sample_size = 3
    overlap = 0
    lines =
      (rows: 10, columns: 7)
      [[ 0.  1. ...  5.  6.]
       [ 7.  8. ... 12. 13.]
       ...
       [56. 57. ... 61. 62.]
       [63. 64. ... 68. 69.]]
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    attrs = {}

    Access samples using subscript notation. Notice how the samples are
    consecutive, as `overlap` is 0 by default.

    >>> line_data[0]
    pept.LineData (samples: 1)
    --------------------------
    sample_size = 3
    overlap = 0
    lines =
      (rows: 3, columns: 7)
      [[ 0.  1. ...  5.  6.]
       [ 7.  8. ... 12. 13.]
       [14. 15. ... 19. 20.]]
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    attrs = {}

    >>> line_data[1]
    pept.LineData (samples: 1)
    --------------------------
    sample_size = 3
    overlap = 0
    lines =
      (rows: 3, columns: 7)
      [[21. 22. ... 26. 27.]
       [28. 29. ... 33. 34.]
       [35. 36. ... 40. 41.]]
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    attrs = {}

    Now set an overlap of 2; notice how the number of samples changes:

    >>> len(line_data)     # Number of samples
    3

    >>> line_data.overlap = 2
    >>> len(line_data)
    8

    Notice how rows are repeated from one sample to the next when accessing
    them, because `overlap` is now 2:

    >>> line_data[0]
    pept.LineData (samples: 1)
    --------------------------
    sample_size = 3
    overlap = 0
    lines =
      (rows: 3, columns: 7)
      [[ 0.  1. ...  5.  6.]
       [ 7.  8. ... 12. 13.]
       [14. 15. ... 19. 20.]]
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    attrs = {}

    >>> line_data[1]
    pept.LineData (samples: 1)
    --------------------------
    sample_size = 3
    overlap = 0
    lines =
      (rows: 3, columns: 7)
      [[ 7.  8. ... 12. 13.]
       [14. 15. ... 19. 20.]
       [21. 22. ... 26. 27.]]
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    attrs = {}

    Now change `sample_size` to 5 and notice again how the number of samples
    changes:

    >>> len(line_data)
    8

    >>> line_data.sample_size = 5
    >>> len(line_data)
    2

    >>> line_data[0]
    pept.LineData (samples: 1)
    --------------------------
    sample_size = 5
    overlap = 0
    lines =
      (rows: 5, columns: 7)
      [[ 0.  1. ...  5.  6.]
       [ 7.  8. ... 12. 13.]
       ...
       [21. 22. ... 26. 27.]
       [28. 29. ... 33. 34.]]
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    attrs = {}

    >>> line_data[1]
    pept.LineData (samples: 1)
    --------------------------
    sample_size = 5
    overlap = 0
    lines =
      (rows: 5, columns: 7)
      [[21. 22. ... 26. 27.]
       [28. 29. ... 33. 34.]
       ...
       [42. 43. ... 47. 48.]
       [49. 50. ... 54. 55.]]
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    attrs = {}

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
            Any extra attributes to set in `.attrs`.

        Raises
        ------
        ValueError
            If `lines` has fewer than 7 columns.

        ValueError
            If `overlap` >= `sample_size` unless `sample_size` is 0. Overlap
            has to be smaller than `sample_size`. Note that it can also be
            negative.
        '''

        # Copy-constructor
        if isinstance(lines, LineData):
            kwargs.update(lines.attrs)
            columns = lines.columns
            lines = lines.lines.copy()

        # Iterable of LineData
        if len(lines) and isinstance(lines[0], LineData):
            kwargs.update(lines[0].attrs)
            columns = lines[0].columns
            sample_size = [len(li.lines) for li in lines]
            lines = np.vstack([li.lines for li in lines])

        # NumPy array-like
        else:
            lines = np.asarray(lines, order = 'C', dtype = float)

        # Check that lines has at least 7 columns.
        if lines.ndim != 2 or lines.shape[1] < 7:
            raise ValueError((
                "\n[ERROR]: `lines` should have dimensions (M, N), where "
                f"N >= 7. Received {lines.shape}.\n"
            ))

        # Call the IterableSamples constructor to make the class iterable in
        # samples with overlap.
        IterableSamples.__init__(
            self,
            lines,
            sample_size,
            overlap,
            columns = columns,
            **kwargs,
        )


    @property
    def lines(self):
        # The `data` attribute is set by the parent class, `IterableSamples`
        return self.data


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


    def __repr__(self):
        # String representation of the class
        name = f"pept.LineData (samples: {len(self)})"
        underline = "-" * len(name)

        # Custom printing of the .lines and .samples_indices arrays
        with np.printoptions(threshold = 5, edgeitems = 2):
            lines_str = f"{indent(str(self.lines), '  ')}"

            if self.sample_size is None:
                samples_indices_str = str(self.samples_indices)
                samples_indices_str = (
                    f"samples_indices = \n"
                    f"{indent(samples_indices_str, '  ')}\n"
                )
            else:
                samples_indices_str = ""

        # Pretty-printing extra attributes
        attrs_str = ""
        if self.attrs:
            items = []
            for k, v in self.attrs.items():
                s = f"  {k.__repr__()}: {v}"
                if len(s) > 75:
                    s = s[:72] + "..."
                items.append(s)
            attrs_str = "\n" + "\n".join(items) + "\n"

        # Return constructed string
        return (
            f"{name}\n{underline}\n"
            f"sample_size = {self.sample_size}\n"
            f"overlap = {self.overlap}\n"
            f"{samples_indices_str}"
            f"lines = \n"
            f"  (rows: {len(self.lines)}, columns: {len(self.columns)})\n"
            f"{lines_str}\n"
            f"columns = {self.columns}\n"
            "attrs = {"
            f"{attrs_str}"
            "}\n"
        )
