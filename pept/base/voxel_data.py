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


# File   : voxel_data.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 07.01.2020


import  pickle
import  time
import  textwrap

import  numpy                   as      np

import  plotly.graph_objects    as      go
import  matplotlib
import  matplotlib.pyplot       as      plt

from    pept.utilities.traverse import  traverse3d

from    .iterable_samples       import  AsyncIterableSamples, PEPTObject
from    .line_data              import  LineData




class Voxels(PEPTObject, np.ndarray):
    '''A class that manages a single 3D voxel space, including tools for voxel
    traversal of lines, manipulation and visualisation.

    This class can be instantiated in a couple of ways:

    1. The constructor receives a pre-defined voxel space (i.e. a 3D numpy
       array), along with the space boundaries `xlim`, `ylim` and `zlim`.

    2. The `from_lines` method receives a sample of 3D lines (i.e. a 2D numpy
       array), each defined by two points, creating a voxel space and
       traversing / voxellising the lines.

    3. The `empty` method creates a voxel space filled with zeros.

    This subclasses the `numpy.ndarray` class, so any `Voxels` object acts
    exactly like a 3D numpy array. All numpy methods and operations are valid
    on `Voxels` (e.g. add 1 to all voxels with `voxels += 1`).

    It is possible to add multiple samples of lines to the same voxel space
    using the `add_lines` method.

    If you want to voxellise multiple samples of lines, see the
    ``pept.tracking.Voxelize`` class.

    Attributes
    ----------
    voxels: (M, N, P) numpy.ndarray
        The 3D numpy array containing the number of lines that pass through
        each voxel. They are stored as `float`s. This class assumes a uniform
        grid of voxels - that is, the voxel size in each dimension is constant,
        but can vary from one dimension to another. The number of voxels in
        each dimension is defined by `number_of_voxels`.

    number_of_voxels: 3-tuple
        A 3-tuple corresponding to the shape of `voxels`.

    voxel_size: (3,) numpy.ndarray
        The lengths of a voxel in the x-, y- and z-dimensions, respectively.

    xlim: (2,) numpy.ndarray
        The lower and upper boundaries of the voxellised volume in the
        x-dimension, formatted as [x_min, x_max].

    ylim: (2,) numpy.ndarray
        The lower and upper boundaries of the voxellised volume in the
        y-dimension, formatted as [y_min, y_max].

    zlim: (2,) numpy.ndarray
        The lower and upper boundaries of the voxellised volume in the
        z-dimension, formatted as [z_min, z_max].

    voxel_grids: list[numpy.ndarray]
        A list containing the voxel gridlines in the x-, y-, and z-dimensions.
        Each dimension's gridlines are stored as a numpy of the voxel
        delimitations, such that it has length (M + 1), where M is the number
        of voxels in given dimension.

    Notes
    -----
    The traversed lines do not need to be fully bounded by the voxel space.
    Their intersection is automatically computed.

    The class saves `voxels` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the shape of the array after instantiating the class.

    Examples
    --------
    This class is most often instantiated from a sample of lines to voxellise:

    >>> import pept
    >>> import numpy as np

    >>> lines = np.arange(70).reshape(10, 7)

    >>> number_of_voxels = [3, 4, 5]
    >>> voxels = pept.Voxels.from_lines(lines, number_of_voxels)
    >>> Initialised Voxels class in 0.0006861686706542969 s.

    >>> print(voxels)
    >>> voxels:
    >>> [[[2. 1. 0. 0. 0.]
    >>>   [0. 2. 0. 0. 0.]
    >>>   [0. 0. 0. 0. 0.]
    >>>   [0. 0. 0. 0. 0.]]

    >>>  [[0. 0. 0. 0. 0.]
    >>>   [0. 1. 1. 0. 0.]
    >>>   [0. 0. 1. 1. 0.]
    >>>   [0. 0. 0. 0. 0.]]

    >>>  [[0. 0. 0. 0. 0.]
    >>>   [0. 0. 0. 0. 0.]
    >>>   [0. 0. 0. 2. 0.]
    >>>   [0. 0. 0. 1. 2.]]]

    >>> number_of_voxels =    (3, 4, 5)
    >>> voxel_size =          [22.  16.5 13.2]

    >>> xlim =                [ 1. 67.]
    >>> ylim =                [ 2. 68.]
    >>> zlim =                [ 3. 69.]

    >>> voxel_grids:
    >>> [array([ 1., 23., 45., 67.]),
    >>>  array([ 2. , 18.5, 35. , 51.5, 68. ]),
    >>>  array([ 3. , 16.2, 29.4, 42.6, 55.8, 69. ])]

    Note that it is important to define the `number_of_voxels`.

    See Also
    --------
    pept.VoxelData : Asynchronously manage multiple voxel spaces.
    pept.LineData : Encapsulate lines for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    '''

    def __new__(
        cls,
        voxels_array,
        xlim,
        ylim,
        zlim,
    ):
        '''`Voxels` class constructor.

        Parameters
        ----------
        voxels_array: 3D numpy.ndarray
            A 3D numpy array, corresponding to a pre-defined voxel space.

        xlim: (2,) numpy.ndarray
            The lower and upper boundaries of the voxellised volume in the
            x-dimension, formatted as [x_min, x_max].

        ylim: (2,) numpy.ndarray
            The lower and upper boundaries of the voxellised volume in the
            y-dimension, formatted as [y_min, y_max].

        zlim: (2,) numpy.ndarray
            The lower and upper boundaries of the voxellised volume in the
            z-dimension, formatted as [z_min, z_max].

        Raises
        ------
        ValueError
            If `voxels_array` does not have exactly 3 dimensions or if
            `xlim`, `ylim` or `zlim` do not have exactly 2 values each.

        '''

        # Type-checking inputs
        voxels_array = np.asarray(
            voxels_array,
            order = "C",
            dtype = float
        )

        if voxels_array.ndim != 3:
            raise ValueError(textwrap.fill((
                "The input `voxels_array` must contain an array-like with "
                "exactly three dimensions (i.e. pre-made voxels array). "
                f"Received an array with {voxels_array.ndim} dimensions. "
                "Note: if you would like to create voxels from a sample of"
                "lines, use the `Voxels.from_lines` method. "
            )))

        xlim = np.asarray(xlim, dtype = float)

        if xlim.ndim != 1 or len(xlim) != 2:
            raise ValueError(textwrap.fill((
                "The input `xlim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the voxel space in the x-dimension. "
                f"Received parameter with shape {xlim.shape}."
            )))

        ylim = np.asarray(ylim, dtype = float)

        if ylim.ndim != 1 or len(ylim) != 2:
            raise ValueError(textwrap.fill((
                "The input `ylim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the voxel space in the y-dimension. "
                f"Received parameter with shape {ylim.shape}."
            )))

        zlim = np.asarray(zlim, dtype = float)

        if zlim.ndim != 1 or len(zlim) != 2:
            raise ValueError(textwrap.fill((
                "The input `zlim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the voxel space in the z-dimension. "
                f"Received parameter with shape {zlim.shape}."
            )))

        # Setting class attributes
        voxels = voxels_array.view(cls)
        voxels._number_of_voxels = voxels.shape

        voxels._xlim = xlim
        voxels._ylim = ylim
        voxels._zlim = zlim

        voxels._voxel_size = np.array([
            (voxels._xlim[1] - voxels._xlim[0]) / voxels._number_of_voxels[0],
            (voxels._ylim[1] - voxels._ylim[0]) / voxels._number_of_voxels[1],
            (voxels._zlim[1] - voxels._zlim[0]) / voxels._number_of_voxels[2],
        ])

        voxels._voxel_grids = tuple([
            np.linspace(lim[0], lim[1], voxels._number_of_voxels[i] + 1)
            for i, lim in enumerate((voxels._xlim, voxels._ylim, voxels._zlim))
        ])

        voxels._attrs = dict()

        return voxels


    def __array_finalize__(self, voxels):
        if voxels is None:
            return

        self._number_of_voxels = getattr(voxels, "_number_of_voxels", None)
        self._voxel_size = getattr(voxels, "_voxel_size", None)

        self._xlim = getattr(voxels, "_xlim", None)
        self._ylim = getattr(voxels, "_ylim", None)
        self._zlim = getattr(voxels, "_zlim", None)

        self._voxel_grids = getattr(voxels, "_voxel_grids", None)
        self._attrs = getattr(voxels, "_attrs", None)


    def __reduce__(self):
        # __reduce__ and __setstate__ ensure correct pickling behaviour. See
        # https://stackoverflow.com/questions/26598109/preserve-custom-
        # attributes-when-pickling-subclass-of-numpy-array

        # Get the parent's __reduce__ tuple
        pickled_state = super(Voxels, self).__reduce__()

        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (
            self._number_of_voxels,
            self._voxel_size,
            self._xlim,
            self._ylim,
            self._zlim,
            self._voxel_grids,
            self._attrs,
        )

        # Return a tuple that replaces the parent's __setstate__ tuple with
        # our own
        return (pickled_state[0], pickled_state[1], new_state)


    def __setstate__(self, state):
        # __reduce__ and __setstate__ ensure correct pickling behaviour
        # https://stackoverflow.com/questions/26598109/preserve-custom-
        # attributes-when-pickling-subclass-of-numpy-array

        # Set the class attributes
        self._attrs = state[-1]
        self._voxel_grids = state[-2]
        self._zlim = state[-3]
        self._ylim = state[-4]
        self._xlim = state[-5]
        self._voxel_size = state[-6]
        self._number_of_voxels = state[-7]

        # Call the parent's __setstate__ with the other tuple elements.
        super(Voxels, self).__setstate__(state[0:-7])


    @property
    def voxels(self):
        return self.__array__()


    @property
    def number_of_voxels(self):
        return self._number_of_voxels


    @property
    def xlim(self):
        return self._xlim


    @property
    def ylim(self):
        return self._ylim


    @property
    def zlim(self):
        return self._zlim


    @property
    def voxel_size(self):
        return self._voxel_size


    @property
    def voxel_grids(self):
        return self._voxel_grids


    @property
    def attrs(self):
        return self._attrs


    @staticmethod
    def from_lines(
        lines,
        number_of_voxels,
        xlim = None,
        ylim = None,
        zlim = None,
        verbose = True,
    ):
        '''Create a voxel space and traverse / voxellise a given sample of
        `lines`.

        The `number_of_voxels` in each dimension must be defined. If the
        voxel space boundaries `xlim`, `ylim` or `zlim` are not defined, they
        are inferred as the boundaries of the `lines`.

        Parameters
        ----------
        lines : (M, N>=7) numpy.ndarray or pept.LineData
            The lines that will be voxellised, each defined by a timestamp and
            two 3D points, so that the data columns are [time, x1, y1, z1,
            x2, y2, z2, ...]. Note that extra columns are ignored.

        number_of_voxels : (3,) list[int]
            The number of voxels in the x-, y-, and z-dimensions, respectively.

        xlim : (2,) list[float], optional
            The lower and upper boundaries of the voxellised volume in the
            x-dimension, formatted as [x_min, x_max]. If undefined, it is
            inferred from the boundaries of `lines`.

        ylim : (2,) list[float], optional
            The lower and upper boundaries of the voxellised volume in the
            y-dimension, formatted as [y_min, y_max]. If undefined, it is
            inferred from the boundaries of `lines`.

        zlim : (2,) list[float], optional
            The lower and upper boundaries of the voxellised volume in the
            z-dimension, formatted as [z_min, z_max]. If undefined, it is
            inferred from the boundaries of `lines`.

        Returns
        -------
        pept.Voxels
            A new `Voxels` object with the voxels through which the lines were
            traversed.

        Raises
        ------
        ValueError
            If the input `lines` does not have the shape (M, N>=7). If the
            `number_of_voxels` is not a 1D list with exactly 3 elements, or
            any dimension has fewer than 2 voxels.

        '''
        if verbose:
            start = time.time()

        # Type-checking inputs
        if not isinstance(lines, LineData):
            lines = LineData(lines)

        lines = lines.lines

        number_of_voxels = np.asarray(
            number_of_voxels,
            order = "C",
            dtype = int
        )

        if number_of_voxels.ndim != 1 or len(number_of_voxels) != 3:
            raise ValueError(textwrap.fill((
                "The input `number_of_voxels` must be a list-like "
                "with exactly three values, corresponding to the "
                "number of voxels in the x-, y-, and z-dimension. "
                f"Received parameter with shape {number_of_voxels.shape}."
            )))

        if (number_of_voxels < 2).any():
            raise ValueError(textwrap.fill((
                "The input `number_of_voxels` must set at least two "
                "voxels in each dimension (i.e. all elements in "
                "`number_of_elements` must be larger or equal to two). "
                f"Received `{number_of_voxels}`."
            )))

        if xlim is None:
            xlim = Voxels.get_cutoff(lines[:, 1], lines[:, 4])
        else:
            xlim = np.asarray(xlim, dtype = float)

            if xlim.ndim != 1 or len(xlim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `xlim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the voxel space in the x-dimension. "
                    f"Received parameter with shape {xlim.shape}."
                )))

        if ylim is None:
            ylim = Voxels.get_cutoff(lines[:, 2], lines[:, 5])
        else:
            ylim = np.asarray(ylim, dtype = float)

            if ylim.ndim != 1 or len(ylim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `ylim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the voxel space in the y-dimension. "
                    f"Received parameter with shape {ylim.shape}."
                )))

        if zlim is None:
            zlim = Voxels.get_cutoff(lines[:, 3], lines[:, 6])
        else:
            zlim = np.asarray(zlim, dtype = float)

            if zlim.ndim != 1 or len(zlim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `zlim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the voxel space in the z-dimension. "
                    f"Received parameter with shape {ylim.shape}."
                )))

        voxels_array = np.zeros(tuple(number_of_voxels))
        voxels = Voxels(
            voxels_array,
            xlim = xlim,
            ylim = ylim,
            zlim = zlim,
        )

        voxels.add_lines(lines, verbose = False)

        if verbose:
            end = time.time()
            print((
                f"Initialised Voxels class in {end - start} s."
            ))

        return voxels


    @staticmethod
    def empty(number_of_voxels, xlim, ylim, zlim):
        '''Create an empty voxel space for the 3D cube bounded by `xlim`,
        `ylim` and `zlim`.

        Parameters
        ----------
        number_of_voxels: (3,) numpy.ndarray
            A list-like containing the number of voxels to be created in the
            x-, y- and z-dimension, respectively.

        xlim: (2,) numpy.ndarray
            The lower and upper boundaries of the voxellised volume in the
            x-dimension, formatted as [x_min, x_max].

        ylim: (2,) numpy.ndarray
            The lower and upper boundaries of the voxellised volume in the
            y-dimension, formatted as [y_min, y_max].

        zlim: (2,) numpy.ndarray
            The lower and upper boundaries of the voxellised volume in the
            z-dimension, formatted as [z_min, z_max].

        Raises
        ------
        ValueError
            If `number_of_voxels` does not have exactly 3 values, or it has
            values smaller than 2. If `xlim`, `ylim` or `zlim` do not have
            exactly 2 values each.

        '''

        number_of_voxels = np.asarray(
            number_of_voxels,
            order = "C",
            dtype = int
        )

        if number_of_voxels.ndim != 1 or len(number_of_voxels) != 3:
            raise ValueError(textwrap.fill((
                "The input `number_of_voxels` must be a list-like "
                "with exactly three values, corresponding to the "
                "number of voxels in the x-, y-, and z-dimension. "
                f"Received parameter with shape {number_of_voxels.shape}."
            )))

        if (number_of_voxels < 2).any():
            raise ValueError(textwrap.fill((
                "The input `number_of_voxels` must set at least two "
                "voxels in each dimension (i.e. all elements in "
                "`number_of_elements` must be larger or equal to two). "
                f"Received `{number_of_voxels}`."
            )))

        number_of_voxels = tuple(number_of_voxels)
        empty_voxels = np.zeros(number_of_voxels)

        return Voxels(
            empty_voxels,
            xlim = xlim,
            ylim = ylim,
            zlim = zlim,
        )


    @staticmethod
    def get_cutoff(p1, p2):
        '''Return a numpy array containing the minimum and maximum value found
        across the two input arrays.

        Parameters
        ----------
        p1 : (N,) numpy.ndarray
            The first 1D numpy array.

        p2 : (N,) numpy.ndarray
            The second 1D numpy array.

        Returns
        -------
        (2,) numpy.ndarray
            The minimum and maximum value found across `p1` and `p2`.

        Notes
        -----
        The input parameters *must* be numpy arrays, otherwise an error will
        be raised.

        '''

        return np.array([
            min(p1.min(), p2.min()),
            max(p1.max(), p2.max()),
        ])


    def save(self, filepath):
        '''Save a `Voxels` instance as a binary `pickle` object.

        Saves the full object state, including the inner `.voxels` NumPy array,
        `xlim`, etc. in a fast, portable binary format. Load back the object
        using the `load` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Examples
        --------
        Save a `Voxels` instance, then load it back:

        >>> voxels = pept.Voxels.empty((64, 48, 32), [0, 20], [0, 10], [0, 5])
        >>> voxels.save("voxels.pickle")

        >>> voxels_reloaded = pept.Voxels.load("voxels.pickle")

        '''
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(filepath):
        '''Load a saved / pickled `Voxels` object from `filepath`.

        Most often the full object state was saved using the `.save` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Returns
        -------
        pept.Voxels
            The loaded `pept.Voxels` instance.

        Examples
        --------
        Save a `Voxels` instance, then load it back:

        >>> voxels = pept.Voxels.empty((64, 48, 32), [0, 20], [0, 10], [0, 5])
        >>> voxels.save("voxels.pickle")

        >>> voxels_reloaded = pept.Voxels.load("voxels.pickle")

        '''
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        return obj


    def add_lines(self, lines, verbose = False):
        '''Voxellise a sample of lines, adding 1 to each voxel traversed, for
        each line in the sample.

        Parameters
        ----------
        lines : (M, N >= 7) numpy.ndarray
            The sample of 3D lines to voxellise. Each line is defined as a
            timestamp followed by two 3D points, such that the data columns are
            `[time, x1, y1, z1, x2, y2, z2, ...]`. Note that there can be extra
            data columns which will be ignored.

        verbose : bool, default False
            Time the voxel traversal and print it to the terminal.

        Raises
        ------
        ValueError
            If `lines` has fewer than 7 columns.

        '''

        lines = np.asarray(lines, order = "C", dtype = float)
        if lines.ndim != 2 or lines.shape[1] < 7:
            raise ValueError(textwrap.fill((
                "The input `lines` must be a 2D array of lines, where each "
                "line (i.e. row) is defined by a timestamp and two 3D points, "
                "so the data columns are [time, x1, y1, z1, x2, y2, z2]. "
                f"Received array of shape {lines.shape}."
            )))

        if verbose:
            start = time.time()

        traverse3d(
            self.voxels,
            lines,
            self._voxel_grids[0],
            self._voxel_grids[1],
            self._voxel_grids[2]
        )

        if verbose:
            end = time.time()
            print(f"Traversing {len(lines)} lines took {end - start} s.")


    def plot(
        self,
        condition = lambda voxel_data: voxel_data > 0,
        ax = None,
        alt_axes = False,
    ):
        '''Plot the voxels in this class using Matplotlib.

        This plots the centres of all voxels encapsulated in a `pept.Voxels`
        instance, colour-coding the voxel value.

        The `condition` parameter is a filtering function that should return
        a boolean mask (i.e. it is the result of a condition evaluation). For
        example `lambda x: x > 0` selects all voxels that have a value larger
        than 0.

        Parameters
        ----------
        condition : function, default `lambda voxel_data: voxel_data > 0`
            The filtering function applied to the voxel data before plotting
            it. It should return a boolean mask (a numpy array of the same
            shape, filled with True and False), selecting all voxels that
            should be plotted. The default, `lambda x: x > 0` selects all
            voxels which have a value larger than 0.

        ax : mpl_toolkits.mplot3D.Axes3D object, optional
            The 3D matplotlib-based axis for plotting. If undefined, new
            Matplotlib figure and axis objects are created.

        alt_axes : bool, default False
            If `True`, plot using the alternative PEPT-style axes convention:
            z is horizontal, y points upwards. Because Matplotlib cannot swap
            axes, this is achieved by swapping the parameters in the plotting
            call (i.e. `plt.plot(x, y, z)` -> `plt.plot(z, x, y)`).

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
        Voxellise an array of lines and add them to a `PlotlyGrapher` instance:

        >>> lines = np.array(...)           # shape (N, M >= 7)
        >>> number_of_voxels = [10, 10, 10]
        >>> voxels = pept.Voxels(lines, number_of_voxels)

        >>> fig, ax = voxels.plot()
        >>> fig.show()

        '''

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
        else:
            fig = plt.gcf()

        filtered_indices = np.argwhere(condition(self))
        positions = self._voxel_size * (0.5 + filtered_indices) + \
            [self._xlim[0], self._ylim[0], self._zlim[0]]

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        voxel_vals = np.array([self[tuple(fi)] for fi in filtered_indices])

        cmap = plt.cm.magma
        color_array = cmap(voxel_vals / voxel_vals.max())

        if alt_axes:
            ax.scatter(z, x, y, c = color_array, marker = "s")

            ax.set_xlabel("z (mm)")
            ax.set_ylabel("x (mm)")
            ax.set_zlabel("y (mm)")

        else:
            ax.scatter(x, y, z, c = color_array, marker = "s")

            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_zlabel("z (mm)")

        return fig, ax


    def cube_trace(
        self,
        index,
        color = None,
        opacity = 0.4,
        colorbar = True,
        colorscale = "magma",
    ):
        '''Get the Plotly `Mesh3d` trace for a single voxel at `index`.

        This renders the voxel as a cube. While visually accurate, this method
        is *very* computationally intensive - only use it for fewer than 100
        cubes. For more voxels, use the `voxels_trace` method.

        Parameters
        ----------
        index: (3,) tuple
            The voxel indices, given as a 3-tuple.

        color : str or list-like, optional
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Overrides `colorbar` if set. For more information,
            check the Plotly documentation. The default is None.

        opacity : float, default 0.4
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque.

        colorbar : bool, default True
            If set to True, will color-code the voxel values. Is overridden if
            `color` is set.

        colorscale : str, default "Magma"
            The Plotly scheme for color-coding the voxel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        Raises
        ------
        ValueError
            If `index` does not contain exactly three values.

        Notes
        -----
        If you want to render a small number of voxels as cubes using Plotly,
        use the `cubes_traces` method, which creates a list of individual cubes
        for all voxels, using this function.

        '''

        index = np.asarray(index, dtype = int)

        if index.ndim != 1 or len(index) != 3:
            raise ValueError(textwrap.fill((
                "The input `index` must contain exactly three values, "
                "corresponding to the x, y, z indices of the voxel to plot. "
                f"Received {index}."
            )))

        xyz = self._voxel_size * index + \
            [self._xlim[0], self._ylim[0], self._zlim[0]]

        x = np.array([0, 0, 1, 1, 0, 0, 1, 1]) * self._voxel_size[0]
        y = np.array([0, 1, 1, 0, 0, 1, 1, 0]) * self._voxel_size[1]
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * self._voxel_size[2]
        i = np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2])
        j = np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3])
        k = np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])

        cube = dict(
            x =  x + xyz[0],
            y =  y + xyz[1],
            z =  z + xyz[2],
            i =  i,
            j =  j,
            k =  k,
            opacity = opacity,
            color = color
        )

        if colorbar and color is None:
            cmap = matplotlib.cm.get_cmap(colorscale)
            c = cmap(self[tuple(index)] / (self.max() or 1))
            cube.update(
                color = "rgb({},{},{})".format(c[0], c[1], c[2])
            )

        return go.Mesh3d(cube)


    def cubes_traces(
        self,
        condition = lambda voxels: voxels > 0,
        color = None,
        opacity = 0.4,
        colorbar = True,
        colorscale = "magma",
    ):
        '''Get a list of Plotly `Mesh3d` traces for all voxels selected by the
        `condition` filtering function.

        The `condition` parameter is a filtering function that should return
        a boolean mask (i.e. it is the result of a condition evaluation). For
        example `lambda x: x > 0` selects all voxels that have a value larger
        than 0.

        This renders each voxel as individual cubes. While visually accurate,
        this method is *very* computationally intensive - only use it for fewer
        than 100 cubes. For more voxels, use the `voxels_trace` method.

        Parameters
        ----------
        condition : function, default `lambda voxels: voxels > 0`
            The filtering function applied to the voxel data before plotting
            it. It should return a boolean mask (a numpy array of the same
            shape, filled with True and False), selecting all voxels that
            should be plotted. The default, `lambda x: x > 0` selects all
            voxels which have a value larger than 0.

        color : str or list-like, optional
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Overrides `colorbar` if set. For more information,
            check the Plotly documentation. The default is None.

        opacity : float, default 0.4
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque.

        colorbar : bool, default True
            If set to True, will color-code the voxel values. Is overridden if
            `color` is set.

        colorscale : str, default "magma"
            The Plotly scheme for color-coding the voxel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        Examples
        --------
        Voxellise an array of lines and add them to a `PlotlyGrapher` instance:

        >>> grapher = PlotlyGrapher()
        >>> lines = np.array(...)           # shape (N, M >= 7)

        >>> number_of_voxels = [10, 10, 10]
        >>> voxels = pept.Voxels(lines, number_of_voxels)

        >>> grapher.add_lines(lines)
        >>> grapher.add_traces(voxels.cubes_traces())  # small number of voxels
        >>> grapher.show()

        '''

        indices = np.argwhere(condition(self))
        traces = [
            self.cube_trace(
                i,
                color = color,
                opacity = opacity,
                colorbar = colorbar,
                colorscale = colorscale,
            ) for i in indices
        ]

        return traces


    def voxels_trace(
        self,
        condition = lambda voxel_data: voxel_data > 0,
        size = 4,
        color = None,
        opacity = 0.4,
        colorbar = True,
        colorscale = "Magma",
        colorbar_title = None,
    ):
        '''Create and return a trace for all the voxels in this class, with
        possible filtering.

        Creates a `plotly.graph_objects.Scatter3d` object for the centres of
        all voxels encapsulated in a `pept.Voxels` instance, colour-coding the
        voxel value.

        The `condition` parameter is a filtering function that should return
        a boolean mask (i.e. it is the result of a condition evaluation). For
        example `lambda x: x > 0` selects all voxels that have a value larger
        than 0.

        Parameters
        ----------
        condition : function, default `lambda voxel_data: voxel_data > 0`
            The filtering function applied to the voxel data before plotting
            it. It should return a boolean mask (a numpy array of the same
            shape, filled with True and False), selecting all voxels that
            should be plotted. The default, `lambda x: x > 0` selects all
            voxels which have a value larger than 0.

        size : float, default 4
            The size of the plotted voxel points. Note that due to the large
            number of voxels in typical applications, the *voxel centres* are
            plotted as square points, which provides an easy to understand
            image that is also fast and responsive.

        color : str or list-like, optional
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Overrides `colorbar` if set. For more information,
            check the Plotly documentation. The default is None.

        opacity : float, default 0.4
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque.

        colorbar : bool, default True
            If set to True, will color-code the voxel values. Is overridden if
            `color` is set.

        colorscale : str, default "Magma"
            The Plotly scheme for color-coding the voxel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        colorbar_title : str, optional
            If set, the colorbar will have this title above it.

        Examples
        --------
        Voxellise an array of lines and add them to a `PlotlyGrapher` instance:

        >>> grapher = PlotlyGrapher()
        >>> lines = np.array(...)           # shape (N, M >= 7)
        >>> number_of_voxels = [10, 10, 10]
        >>> voxels = pept.Voxels.from_lines(lines, number_of_voxels)
        >>> grapher.add_lines(lines)
        >>> grapher.add_trace(voxels.voxels_trace())
        >>> grapher.show()

        '''

        filtered_indices = np.argwhere(condition(self))
        positions = self._voxel_size * (0.5 + filtered_indices) + \
            [self._xlim[0], self._ylim[0], self._zlim[0]]

        marker = dict(
            size = size,
            color = color,
            symbol = "square",
        )

        if colorbar and color is None:
            voxel_vals = [self[tuple(fi)] for fi in filtered_indices]
            marker.update(colorscale = "Magma", color = voxel_vals)

            if colorbar_title is not None:
                marker.update(colorbar = dict(title = colorbar_title))

        voxels = dict(
            x = positions[:, 0],
            y = positions[:, 1],
            z = positions[:, 2],
            opacity = opacity,
            mode = "markers",
            marker = marker,
        )

        return go.Scatter3d(voxels)


    def heatmap_trace(
        self,
        ix = None,
        iy = None,
        iz = None,
        width = 0,
        colorscale = "Magma",
        transpose = True
    ):
        '''Create and return a Plotly `Heatmap` trace of a 2D slice through the
        voxels.

        The orientation of the slice is defined by the input `ix` (for the YZ
        plane), `iy` (XZ), `iz` (XY) parameters - which correspond to the
        voxel index in the x-, y-, and z-dimension. Importantly, at least one
        of them must be defined.

        Parameters
        ----------
        ix : int, optional
            The index along the x-axis of the voxels at which a YZ slice is to
            be taken. One of `ix`, `iy` or `iz` must be defined.

        iy: int, optional
            The index along the y-axis of the voxels at which a XZ slice is to
            be taken. One of `ix`, `iy` or `iz` must be defined.

        iz : int, optional
            The index along the z-axis of the voxels at which a XY slice is to
            be taken. One of `ix`, `iy` or `iz` must be defined.

        width : int, default 0
            The number of voxel layers around the given slice index to collapse
            (i.e. accumulate) onto the heatmap.

        colorscale : str, default "Magma"
            The Plotly scheme for color-coding the voxel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        transpose : bool, default True
            Transpose the heatmap (i.e. flip it across its diagonal).

        Raises
        ------
        ValueError
            If neither of `ix`, `iy` or `iz` was defined.

        Examples
        --------
        Voxellise an array of lines and add them to a `PlotlyGrapher` instance:

        >>> lines = np.array(...)           # shape (N, M >= 7)
        >>> number_of_voxels = [10, 10, 10]
        >>> voxels = pept.Voxels(lines, number_of_voxels)

        >>> import plotly.graph_objs as go
        >>> fig = go.Figure()
        >>> fig.add_trace(voxels.heatmap_trace())
        >>> fig.show()

        '''

        if ix is not None:
            x = self._voxel_grids[1]
            y = self._voxel_grids[2]
            z = self[ix, :, :]

            for i in range(1, width + 1):
                z = z + self[ix + i, :, :]
                z = z + self[ix - i, :, :]

        elif iy is not None:
            x = self._voxel_grids[0]
            y = self._voxel_grids[2]
            z = self[:, iy, :]

            for i in range(1, width + 1):
                z = z + self[:, iy + i, :]
                z = z + self[:, iy - i, :]

        elif iz is not None:
            x = self._voxel_grids[0]
            y = self._voxel_grids[1]
            z = self[:, :, iz]

            for i in range(1, width + 1):
                z = z + self[:, :, iz + i]
                z = z + self[:, :, iz - i]

        else:
            raise ValueError(textwrap.fill((
                "[ERROR]: One of the `ix`, `iy`, `iz` slice indices must be "
                "provided."
            )))

        heatmap = dict(
            x = x,
            y = y,
            z = z,
            colorscale = colorscale,
            transpose = transpose,
        )

        return go.Heatmap(heatmap)


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            "Voxels\n------\n"
            f"{self.__array__()}\n\n"
            f"number_of_voxels =    {self._number_of_voxels}\n"
            f"voxel_size =          {self._voxel_size}\n\n"
            f"xlim =                {self._xlim}\n"
            f"ylim =                {self._ylim}\n"
            f"zlim =                {self._zlim}\n\n"
            f"voxel_grids:\n"
            f"([{self._voxel_grids[0][0]} ... {self._voxel_grids[0][-1]}],\n"
            f" [{self._voxel_grids[1][0]} ... {self._voxel_grids[1][-1]}],\n"
            f" [{self._voxel_grids[2][0]} ... {self._voxel_grids[2][-1]}])"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPR
        return self.__str__()




class VoxelData(AsyncIterableSamples):
    '''A class that can voxellise multiple samples of lines (from a `LineData`)
    lazily / on demand.

    Voxellisation is a computationally-intensive step - in terms of both time
    and memory - especially for many thousands of lines or samples; although
    optimised, the `Voxels` class only manages a single voxel space. The simple
    solution for multiple samples:

    .. code-block:: python

        lines = pept.LineData(...)
        voxels = []

        for sample in lines:
            voxels.append( pept.Voxels(sample, (100, 100, 100)) )


    Is very inefficient, as it uses a single thread and stores all voxel
    spaces in memory (voxels use up a lot of memory!). This class solves these
    problems by:

    1. Voxellising the samples of lines in parallel, on any number of threads.
    2. Creating the voxel spaces on demand. Optionally, it can save / cache
       the expensive voxellisation steps (`save_cache = True`).

    The individual voxellisation steps are still done using `Voxels`, which are
    then accessible in a list (`voxels.voxels` or `voxels[0]`).

    Attributes
    ----------
    line_data : pept.LineData
        The samples of lines encapsulated in a `pept.LineData` that will be
        used to create the corresponding voxellised spaces.

    voxels : list[pept.Voxels | None]
        The list of cached voxellised spaces. Initially, it is a list of
        `None`s of the same length as the number of samples in `line_data`. If
        `save_cache` is True, the voxellised samples of lines will be cached
        here.

    number_of_voxels : 3-tuple
        A 3-tuple corresponding to the shape of `voxels`.

    xlim : (2,) numpy.ndarray
        The lower and upper boundaries of the voxellised volume in the
        x-dimension, formatted as [x_min, x_max].

    ylim : (2,) numpy.ndarray
        The lower and upper boundaries of the voxellised volume in the
        y-dimension, formatted as [y_min, y_max].

    zlim : (2,) numpy.ndarray
        The lower and upper boundaries of the voxellised volume in the
        z-dimension, formatted as [z_min, z_max].

    max_workers : int
        The number of threads that will be used to voxellise the lines in
        parallel.

    save_cache : bool
        Whether to cache the voxellised spaces *when their computation is
        requested* (e.g. when calling `traverse()`). The `voxels` attribute is
        only populated if this is `True`.

    Methods
    -------
    save(filepath)
        Save a `VoxelData` instance as a binary `pickle` object.

    load(filepath)
        Load a saved / pickled `VoxelData` object from `filepath`.

    traverse(sample_indices = ..., verbose = True)
        Voxellise the samples in `line_data` at indices `samples_indices`.

    accumulate(sample_indices = ..., verbose = True):
        Superimpose the voxellised samples in `line_data` at indices
        `samples_indices` into the same voxel space.

    Notes
    -----
    Upon instantiating the class with an instance of `LineData`, a copy is
    made. It is a logic error to change the `sample_size` and `overlap` after
    instantiation; these should remain constant for the `voxels` list to
    remain correct.

    Examples
    --------
    Create a short list of 3D lines and encapsulate them in a `LineData` class
    to simulate multiple samples of lines.

    >>> import pept
    >>> import numpy as np

    >>> lines_raw = np.arange(70).reshape(10, 7)
    >>> lines = pept.LineData(lines_raw, sample_size = 2)

    The `VoxelData` does not voxellise the samples of lines immediately upon
    instantiation by default:

    >>> number_of_voxels = [3, 4, 5]
    >>> voxel_data = pept.VoxelData(lines, number_of_voxels)
    >>> voxel_data.voxels
    >>> [None, None, None, None, None]

    You can iterate through the `VoxelData` and it will voxellise the samples
    on demand:

    >>> for vox in voxel_data:
    >>>     print(vox)          # not cached by default

    This is the most efficient way to use `VoxelData`, as the voxellised
    samples are NOT cached by default; they are deleted after each loop above
    (voxels consume a lot of memory!):

    >>> voxel_data.voxels
    >>> [None, None, None, None, None]

    If you have enough memory to store all the voxel spaces at once, set
    `save_cache` to True when instantiating the class, or afterwards:

    >>> number_of_voxels = [3, 4, 5]
    >>> voxel_data = pept.VoxelData(lines, number_of_voxels, save_cache = True)
    >>> voxel_data.voxels
    >>> [None, None, None, None, None]

    >>> voxel_data.traverse()               # Actually voxellises each sample
    >>> voxel_data.voxels
    >>> [..Voxels.. , ..Voxels.., ...]

    If you voxellised the samples once and `save_cache = True`, the results are
    cached, so new iterations use the pre-computed voxel spaces:

    >>> voxel_data.accumulate()             # Almost instantaneous, uses cache

    See Also
    --------
    pept.Voxels : Manage a single voxellised sample of lines.
    pept.LineData : Encapsulate lines for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.

    '''

    def __init__(
        self,
        line_data,
        number_of_voxels,
        xlim = None,
        ylim = None,
        zlim = None,
        save_cache = False,
        verbose = True,
    ):
        '''`VoxelData` class constructor.

        Parameters
        ----------
        line_data : pept.LineData
            The samples of lines encapsulated in a `pept.LineData` that will be
            used to create the corresponding voxellised spaces.

        number_of_voxels : (3,) list[int]
            The number of voxels in the x-, y-, and z-dimensions, respectively.

        xlim : (2,) list[float], optional
            The lower and upper boundaries of the voxellised volume in the
            x-dimension, formatted as [x_min, x_max]. If undefined, it is
            inferred from the boundaries of the lines in `line_data`.

        ylim : (2,) list[float], optional
            The lower and upper boundaries of the voxellised volume in the
            y-dimension, formatted as [y_min, y_max]. If undefined, it is
            inferred from the boundaries of the lines in `line_data`.

        zlim : (2,) list[float], optional
            The lower and upper boundaries of the voxellised volume in the
            z-dimension, formatted as [z_min, z_max]. If undefined, it is
            inferred from the boundaries of the lines in `line_data`.

        save_cache : bool, default False
            Whether to cache the voxellised spaces *when their computation is
            requested* (e.g. when calling `traverse()`). The `voxels` attribute
            is only populated if this is `True`.

        verbose : bool, default True
            Show extra information as the voxellisation runs.

        Raises
        ------
        TypeError
            If `line_data` is not an instance (or subclass) of `pept.LineData`.

        ValueError
            If `number_of_voxels` does not have exactly 3 values, or it has
            values smaller than 2. If `xlim`, `ylim` or `zlim` do not have
            exactly 2 values each.

        '''

        # Type-checking inputs
        if not isinstance(line_data, LineData):
            raise TypeError(textwrap.fill((
                "The input `line_data` must be an instance (or subclass "
                f"thereof!) of `pept.LineData`. Received {type(line_data)}."
            )))

        number_of_voxels = np.asarray(
            number_of_voxels,
            order = "C",
            dtype = int
        )

        if number_of_voxels.ndim != 1 or len(number_of_voxels) != 3:
            raise ValueError(textwrap.fill((
                "The input `number_of_voxels` must be a list-like "
                "with exactly three values, corresponding to the "
                "number of voxels in the x-, y-, and z-dimension. "
                f"Received parameter with shape {number_of_voxels.shape}."
            )))

        if (number_of_voxels < 2).any():
            raise ValueError(textwrap.fill((
                "The input `number_of_voxels` must set at least two "
                "voxels in each dimension (i.e. all elements in "
                "`number_of_elements` must be larger or equal to two). "
                f"Received `{number_of_voxels}`."
            )))

        # Alias for the whole array of lines in `line_data`
        lines = line_data.lines

        if xlim is None:
            xlim = Voxels.get_cutoff(lines[:, 1], lines[:, 4])
        else:
            xlim = np.asarray(xlim, dtype = float)

            if xlim.ndim != 1 or len(xlim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `xlim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the voxel space in the x-dimension. "
                    f"Received parameter with shape {xlim.shape}."
                )))

        if ylim is None:
            ylim = Voxels.get_cutoff(lines[:, 2], lines[:, 5])
        else:
            ylim = np.asarray(ylim, dtype = float)

            if ylim.ndim != 1 or len(ylim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `ylim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the voxel space in the y-dimension. "
                    f"Received parameter with shape {ylim.shape}."
                )))

        if zlim is None:
            zlim = Voxels.get_cutoff(lines[:, 3], lines[:, 6])
        else:
            zlim = np.asarray(zlim, dtype = float)

            if zlim.ndim != 1 or len(zlim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `zlim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the voxel space in the z-dimension. "
                    f"Received parameter with shape {ylim.shape}."
                )))

        # Setting class attributes
        self._number_of_voxels = tuple(number_of_voxels)
        self._xlim = xlim
        self._ylim = ylim
        self._zlim = zlim

        AsyncIterableSamples.__init__(
            self,
            line_data,
            Voxels.from_lines,
            args = (self.number_of_voxels,),
            kwargs = dict(xlim = self.xlim, ylim = self.ylim,
                          zlim = self.zlim, verbose = False),
            save_cache = save_cache,
            verbose = verbose,
        )


    @property
    def line_data(self):
        # The `samples` attribute is set by the parent class,
        # `AsyncIterableSamples`
        return self.samples


    @property
    def voxels(self):
        return self.processed


    @property
    def number_of_voxels(self):
        return self._number_of_voxels


    @property
    def xlim(self):
        return self._xlim


    @property
    def ylim(self):
        return self._ylim


    @property
    def zlim(self):
        return self._zlim


    def save(self, filepath):
        '''Save a `VoxelData` instance as a binary `pickle` object.

        Saves the full object state, including the inner `.voxels` NumPy array,
        `xlim`, etc. in a fast, portable binary format. Load back the object
        using the `load` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Examples
        --------
        Save a `VoxelData` instance (created from lines), then load it back:

        >>> lines = pept.LineData([[1, 2, 3, 4, 5, 6, 7]])
        >>> voxel_data = pept.VoxelData(lines, (64, 48, 32))
        >>> voxel_data.save("voxel_data.pickle")

        >>> voxel_data_reloaded = pept.VoxelData.load("voxel_data.pickle")

        '''
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(filepath):
        '''Load a saved / pickled `VoxelData` object from `filepath`.

        Most often the full object state was saved using the `.save` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Returns
        -------
        pept.VoxelData
            The loaded `pept.VoxelData` instance.

        Examples
        --------
        Save a `VoxelData` instance (created from lines), then load it back:

        >>> lines = pept.LineData([[1, 2, 3, 4, 5, 6, 7]])
        >>> voxel_data = pept.VoxelData(lines, (64, 48, 32))
        >>> voxel_data.save("voxel_data.pickle")

        >>> voxel_data_reloaded = pept.VoxelData.load("voxel_data.pickle")

        '''
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        return obj


    def copy(self):
        '''Create a deep copy of an instance of this class.'''
        return pickle.loads(pickle.dumps(self))


    def __str__(self):
        # Shown when calling print(class)
        traversed = sum((v is not None for v in self.voxels))
        total = len(self.voxels)

        # Indent LineData docstring with an extra tab
        line_data_str = "    ".join(self.line_data.__str__().splitlines(True))

        docstr = (
            f"voxels:               {traversed} / {total} traversed & cached\n"
            f"number_of_voxels =    {self.number_of_voxels}\n"
            f"xlim =                {self.xlim}\n"
            f"ylim =                {self.ylim}\n"
            f"zlim =                {self.zlim}\n\n"
            f"line_data:\n    {line_data_str}\n\n"
            f"save_cache =          {self.save_cache}\n"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPR
        docstr = (
            "Class instance that inherits from `pept.VoxelData`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n----------\n"
            f"{self.__str__()}"
        )

        return docstr
