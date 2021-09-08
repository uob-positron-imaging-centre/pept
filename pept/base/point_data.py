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


# File   : point_data.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 19.08.2019


from    textwrap                import  indent

import  numpy                   as      np

import  matplotlib.pyplot       as      plt

from    .iterable_samples       import  IterableSamples




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

    sample_size : int, list[int], pept.TimeWindow or None
        Defining the number of points in a sample; if it is an integer, a
        constant number of points are returned per sample. If it is a list of
        integers, sample `i` will have length `sample_size[i]`. If it is a
        `pept.TimeWindow` instance, each sample will span a fixed time window.
        If `None`, custom sample sizes are returned as per the
        `samples_indices` attribute.

    overlap : int, pept.TimeWindow or None
        Defining the overlapping points between consecutive samples. If `int`,
        constant numbers of points are used. If `pept.TimeWindow`, the overlap
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
        A list of strings with the same number of columns as `points`
        containing each column's name.

    attrs : dict[str, Any]
        A dictionary of other attributes saved on this class. Attribute names
        starting with an underscore are considered "hidden".

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size`. Overlap is required to be smaller than
        `sample_size`, unless `sample_size` is 0. Note that it can also be
        negative.

    Notes
    -----
    This class saves `points` as a **C-contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the number of rows or columns after instantiating the class.

    Examples
    --------
    Initialise a `PointData` instance containing 10 points with a `sample_size`
    of 3.

    >>> import numpy as np
    >>> import pept
    >>> points_raw = np.arange(40).reshape(10, 4)
    >>> print(points_raw)
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

    >>> point_data = pept.PointData(points_raw, sample_size = 3)
    >>> point_data
    pept.PointData (samples: 3)
    ---------------------------
    sample_size = 3
    overlap = 0
    points =
      (rows: 10, columns: 4)
      [[ 0.  1.  2.  3.]
       [ 4.  5.  6.  7.]
       ...
       [32. 33. 34. 35.]
       [36. 37. 38. 39.]]
    columns = ['t', 'x', 'y', 'z']
    attrs = {}

    Access samples using subscript notation. Notice how the samples are
    consecutive, as `overlap` is 0 by default.

    >>> point_data[0]
    pept.PointData (samples: 1)
    ---------------------------
    sample_size = 3
    overlap = 0
    points =
      (rows: 3, columns: 4)
      [[ 0.  1.  2.  3.]
       [ 4.  5.  6.  7.]
       [ 8.  9. 10. 11.]]
    columns = ['t', 'x', 'y', 'z']
    attrs = {}

    >>> point_data[1]
    pept.PointData (samples: 1)
    ---------------------------
    sample_size = 3
    overlap = 0
    points =
      (rows: 3, columns: 4)
      [[12. 13. 14. 15.]
       [16. 17. 18. 19.]
       [20. 21. 22. 23.]]
    columns = ['t', 'x', 'y', 'z']
    attrs = {}

    Now set an overlap of 2; notice how the number of samples changes:

    >>> len(point_data)         # Number of samples
    3

    >>> point_data.overlap = 2
    >>> len(point_data)
    8

    Notice how rows are repeated from one sample to the next when accessing
    them, because `overlap` is now 2:

    >>> point_data[0]
    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]])

    >>> point_data[1]
    array([[ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.],
           [12., 13., 14., 15.]])

    Now change `sample_size` to 5 and notice again how the number of samples
    changes:

    >>> len(point_data)
    8

    >>> point_data.sample_size = 5
    >>> len(point_data)
    2

    >>> point_data[0]
    pept.PointData (samples: 1)
    ---------------------------
    sample_size = 3
    overlap = 0
    points =
      (rows: 3, columns: 4)
      [[ 0.  1.  2.  3.]
       [ 4.  5.  6.  7.]
       [ 8.  9. 10. 11.]]
    columns = ['t', 'x', 'y', 'z']
    attrs = {}

    >>> point_data[1]
    pept.PointData (samples: 1)
    ---------------------------
    sample_size = 3
    overlap = 0
    points =
      (rows: 3, columns: 4)
      [[ 4.  5.  6.  7.]
       [ 8.  9. 10. 11.]
       [12. 13. 14. 15.]]
    columns = ['t', 'x', 'y', 'z']
    attrs = {}

    Notice how the samples do not cover the whole input `points_raw` array, as
    the last lines are omitted - think of the `sample_size` and `overlap`. They
    are still inside the inner `points` attribute of `point_data` though:

    >>> point_data.points
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
    pept.read_csv : Fast CSV file reading into numpy arrays.
    pept.plots.PlotlyGrapher :
        Easy, publication-ready plotting of PEPT-oriented data.
    pept.tracking.Cutpoints : Compute cutpoints from `pept.LineData`.
    '''

    def __init__(
        self,
        points,
        sample_size = None,
        overlap = None,
        columns = ["t", "x", "y", "z"],
        **kwargs,
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

        columns : List[str], default ["t", "x", "y", "z"]
            A list of strings corresponding to the column labels in `points`.

        **kwargs : extra keyword arguments
            Any extra attributes to set on the class instance.

        Raises
        ------
        ValueError
            If `line_data` does not have (N, M) shape, where M >= 4.
        '''

        # Copy-constructor
        if isinstance(points, PointData):
            kwargs.update(points.attrs)
            columns = points.columns
            points = points.points.copy()

        # Iterable of PointData
        if len(points) and isinstance(points[0], PointData):
            kwargs.update(points[0].attrs)
            columns = points[0].columns
            sample_size = [len(p.points) for p in points]
            points = np.vstack([p.points for p in points])

        # NumPy array-like
        else:
            points = np.asarray(points, order = 'C', dtype = float)

        # Check that points has at least 4 columns.
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError((
                "\n[ERROR]: `points` should have dimensions (M, N), where "
                f"N >= 4. Received {points.shape}.\n"
            ))

        # Call the IterableSamples constructor to make the class iterable in
        # samples with overlap.
        IterableSamples.__init__(
            self,
            points,
            sample_size,
            overlap,
            columns = columns,
            **kwargs,
        )


    @property
    def points(self):
        # The `data` attribute is set by the parent class, `IterableSamples`
        return self.data


    def to_csv(self, filepath, delimiter = " "):
        '''Write the inner `points` to a CSV file.

        Write all points stored in the class to a CSV file.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        delimiter : str, default " "
            The delimiter used to separate the values in the CSV file.

        '''

        np.savetxt(filepath, self.points, delimiter = delimiter,
                   header = delimiter.join(self.columns))


    def plot(
        self,
        sample_indices = ...,
        ax = None,
        alt_axes = False,
        colorbar_col = -1,
    ):
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
            points. The default is -1 (the last column).

        Returns
        -------
        fig, ax : matplotlib figure and axes objects

        Notes
        -----
        Plotting all points is very computationally-expensive for matplotlib.
        It is recommended to only plot a couple of samples at a time, or use
        the faster `pept.plots.PlotlyGrapher`.

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
            ax = fig.add_subplot(111, projection = '3d')
        else:
            fig = plt.gcf()

        # Check if sample_indices is an iterable collection (list-like)
        # otherwise just "iterate" over the single number or Ellipsis
        if not hasattr(sample_indices, "__iter__"):
            sample_indices = [sample_indices]

        if sample_indices[0] == Ellipsis:
            x = self.points[:, 1],
            y = self.points[:, 2],
            z = self.points[:, 3],
            color = self.points[:, colorbar_col]
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

        if alt_axes:
            ax.scatter(z, x, y, c = color_array)

            ax.set_xlabel("z (mm)")
            ax.set_ylabel("x (mm)")
            ax.set_zlabel("y (mm)")

        else:
            ax.scatter(x, y, z, c = color_array)

            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_zlabel("z (mm)")

        return fig, ax


    def __repr__(self):
        # String representation of the class
        name = f"pept.PointData (samples: {len(self)})"
        underline = "-" * len(name)

        # Custom printing of the .points and .samples_indices arrays
        with np.printoptions(threshold = 5, edgeitems = 2):
            points_str = f"{indent(str(self.points), '  ')}"

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
            f"points = \n"
            f"  (rows: {len(self.points)}, columns: {len(self.columns)})\n"
            f"{points_str}\n"
            f"columns = {self.columns}\n"
            "attrs = {"
            f"{attrs_str}"
            "}\n"
        )
