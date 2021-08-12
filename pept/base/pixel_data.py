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


# File   : pixel_data.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 07.01.2020


import  pickle
import  time
import  textwrap

import  numpy                   as      np

import  plotly.graph_objects    as      go
import  matplotlib.pyplot       as      plt

from    pept.utilities.traverse import  traverse2d
from    .iterable_samples       import  PEPTObject




class Pixels(np.ndarray, PEPTObject):
    '''A class that manages a 2D pixel space, including tools for pixel
    traversal of lines, manipulation and visualisation.

    This class can be instantiated in a couple of ways:

    1. The constructor receives a pre-defined pixel space (i.e. a 2D numpy
       array), along with the space boundaries `xlim` and `ylim`.

    2. The `from_lines` method receives a sample of 2D lines (i.e. a 2D numpy
       array), each defined by two points, creating a pixel space and
       traversing / pixellising the lines.

    3. The `empty` method creates a pixel space filled with zeros.

    This subclasses the `numpy.ndarray` class, so any `Pixels` object acts
    exactly like a 2D numpy array. All numpy methods and operations are valid
    on `Pixels` (e.g. add 1 to all pixels with `pixels += 1`).

    It is possible to add multiple samples of lines to the same pixel space
    using the `add_lines` method.

    Attributes
    ----------
    pixels: (M, N) numpy.ndarray
        The 2D numpy array containing the number of lines that pass through
        each pixel. They are stored as `float`s. This class assumes a uniform
        grid of pixels - that is, the pixel size in each dimension is constant,
        but can vary from one dimension to another. The number of pixels in
        each dimension is defined by `number_of_pixels`.

    number_of_pixels: 2-tuple
        A 2-tuple corresponding to the shape of `pixels`.

    pixel_size: (2,) numpy.ndarray
        The lengths of a pixel in the x- and y-dimensions, respectively.

    xlim: (2,) numpy.ndarray
        The lower and upper boundaries of the pixellised volume in the
        x-dimension, formatted as [x_min, x_max].

    ylim: (2,) numpy.ndarray
        The lower and upper boundaries of the pixellised volume in the
        y-dimension, formatted as [y_min, y_max].

    pixel_grids: list[numpy.ndarray]
        A list containing the pixel gridlines in the x- and y-dimensions.
        Each dimension's gridlines are stored as a numpy of the pixel
        delimitations, such that it has length (M + 1), where M is the number
        of pixels in a given dimension.

    Methods
    -------
    save(filepath)
        Save a `Pixels` instance as a binary `pickle` object.

    load(filepath)
        Load a saved / pickled `Pixels` object from `filepath`.

    from_lines(lines, number_of_pixels, xlim = None, ylim = None, \
               verbose = True)
        Create a pixel space and traverse / pixellise a given sample of
        `lines`.

    empty(number_of_pixels, xlim, ylim, verbose = False)
        Create an empty pixel space for the 2D rectangle bounded by `xlim` and
        `ylim`.

    get_cutoff(p1, p2)
        Return a numpy array containing the minimum and maximum value found
        across the two input arrays.

    add_lines(lines, verbose = False)
        Pixellise a sample of lines, adding 1 to each pixel traversed, for
        each line in the sample.

    cube_trace(index, color = None, opacity = 0.4, colorbar = True,\
               colorscale = "magma")
        Get the Plotly `Mesh3d` trace for a single pixel at `index`.

    cubes_traces(condition = lambda pixels: pixels > 0, color = None,\
                 opacity = 0.4, colorbar = True, colorscale = "magma")
        Get a list of Plotly `Mesh3d` traces for all pixel selected by the
        `condition` filtering function.

    pixels_trace(condition = lambda pixels: pixels > 0, size = 4,\
                 color = None, opacity = 0.4, colorbar = True,\
                 colorscale = "Magma", colorbar_title = None)
        Create and return a trace for all the pixels in this class, with
        possible filtering.

    heatmap_trace(ix = None, iy = None, iz = None, width = 0,\
                  colorscale = "Magma", transpose = True)
        Create and return a Plotly `Heatmap` trace of a 2D slice through the
        voxels.

    Notes
    -----
    The traversed lines do not need to be fully bounded by the pixel space.
    Their intersection is automatically computed.

    The class saves `pixels` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the shape of the array after instantiating the class.

    Examples
    --------
    This class is most often instantiated from a sample of lines to pixellise:

    >>> import pept
    >>> import numpy as np

    >>> lines = np.arange(70).reshape(10, 7)

    >>> number_of_pixels = [3, 4]
    >>> pixels = pept.Pixels.from_lines(lines, number_of_pixels)
    >>> Initialised Pixels class in 0.0006861686706542969 s.

    >>> print(pixels)
    >>> pixels:
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

    >>> number_of_pixels =    (3, 4, 5)
    >>> pixel_size =          [22.  16.5 13.2]

    >>> xlim =                [ 1. 67.]
    >>> ylim =                [ 2. 68.]
    >>> zlim =                [ 3. 69.]

    >>> pixel_grids:
    >>> [array([ 1., 23., 45., 67.]),
    >>>  array([ 2. , 18.5, 35. , 51.5, 68. ]),
    >>>  array([ 3. , 16.2, 29.4, 42.6, 55.8, 69. ])]

    Note that it is important to define the `number_of_pixels`.

    See Also
    --------
    pept.LineData : Encapsulate lines for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    '''

    def __new__(
        cls,
        pixels_array,
        xlim,
        ylim,
    ):
        '''`Pixels` class constructor.

        Parameters
        ----------
        pixels_array: 3D numpy.ndarray
            A 3D numpy array, corresponding to a pre-defined pixel space.

        xlim: (2,) numpy.ndarray
            The lower and upper boundaries of the pixellised volume in the
            x-dimension, formatted as [x_min, x_max].

        ylim: (2,) numpy.ndarray
            The lower and upper boundaries of the pixellised volume in the
            y-dimension, formatted as [y_min, y_max].

        Raises
        ------
        ValueError
            If `pixels_array` does not have exactly 3 dimensions or if
            `xlim` or `ylim` do not have exactly 2 values each.

        '''

        # Type-checking inputs
        pixels_array = np.asarray(
            pixels_array,
            order = "C",
            dtype = float
        )

        if pixels_array.ndim != 2:
            raise ValueError(textwrap.fill((
                "The input `pixels_array` must contain an array-like with "
                "exactly three dimensions (i.e. pre-made pixels array). "
                f"Received an array with {pixels_array.ndim} dimensions. "
                "Note: if you would like to create pixels from a sample of"
                "lines, use the `Pixels.from_lines` method. "
            )))

        xlim = np.asarray(xlim, dtype = float)

        if xlim.ndim != 1 or len(xlim) != 2:
            raise ValueError(textwrap.fill((
                "The input `xlim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the x-dimension. "
                f"Received parameter with shape {xlim.shape}."
            )))

        ylim = np.asarray(ylim, dtype = float)

        if ylim.ndim != 1 or len(ylim) != 2:
            raise ValueError(textwrap.fill((
                "The input `ylim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the y-dimension. "
                f"Received parameter with shape {ylim.shape}."
            )))

        # Setting class attributes
        pixels = pixels_array.view(cls)
        pixels._number_of_pixels = pixels.shape

        pixels._xlim = xlim
        pixels._ylim = ylim

        pixels._pixel_size = np.array([
            (pixels._xlim[1] - pixels._xlim[0]) / pixels._number_of_pixels[0],
            (pixels._ylim[1] - pixels._ylim[0]) / pixels._number_of_pixels[1],
        ])

        pixels._pixel_grids = tuple([
            np.linspace(lim[0], lim[1], pixels._number_of_pixels[i] + 1)
            for i, lim in enumerate((pixels._xlim, pixels._ylim))
        ])

        return pixels


    def __array_finalize__(self, pixels):
        # Required method for numpy subclassing
        if pixels is None:
            return

        self._number_of_pixels = getattr(pixels, "_number_of_pixels", None)
        self._pixel_size = getattr(pixels, "_pixel_size", None)

        self._xlim = getattr(pixels, "_xlim", None)
        self._ylim = getattr(pixels, "_ylim", None)
        self._zlim = getattr(pixels, "_zlim", None)

        self._pixel_grids = getattr(pixels, "_pixel_grids", None)


    def __reduce__(self):
        # __reduce__ and __setstate__ ensure correct pickling behaviour. See
        # https://stackoverflow.com/questions/26598109/preserve-custom-
        # attributes-when-pickling-subclass-of-numpy-array

        # Get the parent's __reduce__ tuple
        pickled_state = super(Pixels, self).__reduce__()

        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (
            self._number_of_pixels,
            self._xlim,
            self._ylim,
            self._pixel_size,
            self._pixel_grids,
        )

        # Return a tuple that replaces the parent's __setstate__ tuple with
        # our own
        return (pickled_state[0], pickled_state[1], new_state)


    def __setstate__(self, state):
        # __reduce__ and __setstate__ ensure correct pickling behaviour
        # https://stackoverflow.com/questions/26598109/preserve-custom-
        # attributes-when-pickling-subclass-of-numpy-array

        # Set the class attributes
        self._pixel_grids = state[-1]
        self._pixel_size = state[-2]
        self._ylim = state[-3]
        self._xlim = state[-4]
        self._number_of_pixels = state[-5]

        # Call the parent's __setstate__ with the other tuple elements.
        super(Pixels, self).__setstate__(state[0:-5])


    @property
    def pixels(self):
        return self.__array__()


    @property
    def number_of_pixels(self):
        return self._number_of_pixels


    @property
    def xlim(self):
        return self._xlim


    @property
    def ylim(self):
        return self._ylim


    @property
    def pixel_size(self):
        return self._pixel_size


    @property
    def pixel_grids(self):
        return self._pixel_grids


    @staticmethod
    def from_lines(
        lines,
        number_of_pixels,
        xlim = None,
        ylim = None,
        verbose = True,
    ):
        '''Create a pixel space and traverse / pixellise a given sample of
        `lines`.

        The `number_of_pixels` in each dimension must be defined. If the
        pixel space boundaries `xlim` or `ylim` are not defined, they
        are inferred as the boundaries of the `lines`.

        Parameters
        ----------
        lines : (M, N>=5) numpy.ndarray
            The lines that will be pixellised, each defined by a timestamp and
            two 2D points, so that the data columns are [time, x1, y1, x2, y2].
            Note that extra columns are ignored.

        number_of_pixels : (2,) list[int]
            The number of pixels in the x- and y-dimensions, respectively.

        xlim : (2,) list[float], optional
            The lower and upper boundaries of the pixellised volume in the
            x-dimension, formatted as [x_min, x_max]. If undefined, it is
            inferred from the boundaries of `lines`.

        ylim : (2,) list[float], optional
            The lower and upper boundaries of the pixellised volume in the
            y-dimension, formatted as [y_min, y_max]. If undefined, it is
            inferred from the boundaries of `lines`.

        Returns
        -------
        pept.Pixels
            A new `Pixels` object with the pixels through which the lines were
            traversed.

        Raises
        ------
        ValueError
            If the input `lines` does not have the shape (M, N>=5). If the
            `number_of_pixels` is not a 1D list with exactly 2 elements, or
            any dimension has fewer than 2 pixels.

        '''
        if verbose:
            start = time.time()

        # Type-checking inputs
        lines = np.asarray(lines, order = "C", dtype = float)

        if lines.ndim != 2 or lines.shape[1] < 5:
            raise ValueError(textwrap.fill((
                "The input `lines` must be a 2D numpy array containing lines "
                "defined by a timestamp and two 2D points, with every row "
                "formatted as [t, x1, y1, x2, y2]. The `lines` must then have "
                f"shape (M, 5). Received array with shape {lines.shape}."
            )))

        number_of_pixels = np.asarray(
            number_of_pixels,
            order = "C",
            dtype = int
        )

        if number_of_pixels.ndim != 1 or len(number_of_pixels) != 2:
            raise ValueError(textwrap.fill((
                "The input `number_of_pixels` must be a list-like "
                "with exactly two values, corresponding to the "
                "number of pixels in the x- and y-dimension. "
                f"Received parameter with shape {number_of_pixels.shape}."
            )))

        if (number_of_pixels < 2).any():
            raise ValueError(textwrap.fill((
                "The input `number_of_pixels` must set at least two "
                "pixels in each dimension (i.e. all elements in "
                "`number_of_elements` must be larger or equal to two). "
                f"Received `{number_of_pixels}`."
            )))

        if xlim is None:
            xlim = Pixels.get_cutoff(lines[:, 1], lines[:, 3])
        else:
            xlim = np.asarray(xlim, dtype = float)

            if xlim.ndim != 1 or len(xlim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `xlim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the pixel space in the x-dimension. "
                    f"Received parameter with shape {xlim.shape}."
                )))

        if ylim is None:
            ylim = Pixels.get_cutoff(lines[:, 1], lines[:, 3])
        else:
            ylim = np.asarray(ylim, dtype = float)

            if ylim.ndim != 1 or len(ylim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `ylim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the pixel space in the y-dimension. "
                    f"Received parameter with shape {ylim.shape}."
                )))

        pixels_array = np.zeros(tuple(number_of_pixels))
        pixels = Pixels(
            pixels_array,
            xlim = xlim,
            ylim = ylim,
        )

        pixels.add_lines(lines, verbose = False)

        if verbose:
            end = time.time()
            print((
                f"Initialised Pixels class in {end - start} s."
            ))

        return pixels


    @staticmethod
    def empty(number_of_pixels, xlim, ylim):
        '''Create an empty pixel space for the 3D cube bounded by `xlim` and
        `ylim`.

        Parameters
        ----------
        number_of_pixels: (2,) numpy.ndarray
            A list-like containing the number of pixels to be created in the
            x- and y-dimension, respectively.

        xlim: (2,) numpy.ndarray
            The lower and upper boundaries of the pixellised volume in the
            x-dimension, formatted as [x_min, x_max].

        ylim: (2,) numpy.ndarray
            The lower and upper boundaries of the pixellised volume in the
            y-dimension, formatted as [y_min, y_max].
            Time the pixellisation step and print it to the terminal.

        Raises
        ------
        ValueError
            If `number_of_pixels` does not have exactly 2 values, or it has
            values smaller than 2. If `xlim` or `ylim` do not have exactly 2
            values each.

        '''

        number_of_pixels = np.asarray(
            number_of_pixels,
            order = "C",
            dtype = int
        )

        if number_of_pixels.ndim != 1 or len(number_of_pixels) != 2:
            raise ValueError(textwrap.fill((
                "The input `number_of_pixels` must be a list-like "
                "with exactly three values, corresponding to the "
                "number of pixels in the x- and y-dimension. "
                f"Received parameter with shape {number_of_pixels.shape}."
            )))

        if (number_of_pixels < 2).any():
            raise ValueError(textwrap.fill((
                "The input `number_of_pixels` must set at least two "
                "pixels in each dimension (i.e. all elements in "
                "`number_of_elements` must be larger or equal to two). "
                f"Received `{number_of_pixels}`."
            )))

        number_of_pixels = tuple(number_of_pixels)
        empty_pixels = np.zeros(number_of_pixels)

        return Pixels(
            empty_pixels,
            xlim = xlim,
            ylim = ylim,
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
        '''Save a `Pixels` instance as a binary `pickle` object.

        Saves the full object state, including the inner `.pixels` NumPy array,
        `xlim`, etc. in a fast, portable binary format. Load back the object
        using the `load` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Examples
        --------
        Save a `Pixels` instance, then load it back:

        >>> pixels = pept.Pixels.empty((640, 480), [0, 20], [0, 10])
        >>> pixels.save("pixels.pickle")

        >>> pixels_reloaded = pept.Pixels.load("pixels.pickle")

        '''
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(filepath):
        '''Load a saved / pickled `Pixels` object from `filepath`.

        Most often the full object state was saved using the `.save` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Returns
        -------
        pept.Pixels
            The loaded `pept.Pixels` instance.

        Examples
        --------
        Save a `Pixels` instance, then load it back:

        >>> pixels = pept.Pixels.empty((640, 480), [0, 20], [0, 10])
        >>> pixels.save("pixels.pickle")

        >>> pixels_reloaded = pept.Pixels.load("pixels.pickle")

        '''
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        return obj


    def add_lines(self, lines, verbose = False):
        '''Pixellise a sample of lines, adding 1 to each pixel traversed, for
        each line in the sample.

        Parameters
        ----------
        lines : (M, N >= 5) numpy.ndarray
            The sample of 2D lines to pixellise. Each line is defined as a
            timestamp followed by two 2D points, such that the data columns are
            `[time, x1, y1, x2, y2, ...]`. Note that there can be extra data
            columns which will be ignored.

        verbose : bool, default False
            Time the pixel traversal and print it to the terminal.

        Raises
        ------
        ValueError
            If `lines` has fewer than 5 columns.

        '''

        lines = np.asarray(lines, order = "C", dtype = float)
        if lines.ndim != 2 or lines.shape[1] < 5:
            raise ValueError(textwrap.fill((
                "The input `lines` must be a 2D array of lines, where each "
                "line (i.e. row) is defined by a timestamp and two 2D points, "
                "so the data columns are [time, x1, y1, x2, y2]. "
                f"Received array of shape {lines.shape}."
            )))

        if verbose:
            start = time.time()

        traverse2d(
            self.pixels,
            lines,
            self._pixel_grids[0],
            self._pixel_grids[1],
        )

        if verbose:
            end = time.time()
            print(f"Traversing {len(lines)} lines took {end - start} s.")


    def pixels_trace(
        self,
        condition = lambda pixels: pixels > 0,
        opacity = 0.9,
        colorscale = "Magma",
    ):
        '''Create and return a trace with all the pixels in this class, with
        possible filtering.

        Creates a `plotly.graph_objects.Surface` object for the centres of
        all pixels encapsulated in a `pept.Pixels` instance, colour-coding the
        pixel value.

        The `condition` parameter is a filtering function that should return
        a boolean mask (i.e. it is the result of a condition evaluation). For
        example `lambda x: x > 0` selects all pixels that have a value larger
        than 0.

        Parameters
        ----------
        condition : function, default `lambda pixels: pixels > 0`
            The filtering function applied to the pixel data before plotting
            it. It should return a boolean mask (a numpy array of the same
            shape, filled with True and False), selecting all pixels that
            should be plotted. The default, `lambda x: x > 0` selects all
            pixels which have a value larger than 0.

        opacity : float, default 0.4
            The opacity of the surface, where 0 is transparent and 1 is fully
            opaque.

        colorscale : str, default "Magma"
            The Plotly scheme for color-coding the voxel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        Examples
        --------
        Pixellise an array of lines and add them to a `PlotlyGrapher` instance:

        >>> grapher = PlotlyGrapher()
        >>> lines = np.array(...)                   # shape (N, M >= 7)
        >>> lines2d = lines[:, [0, 1, 2, 4, 5]]     # select x, y of lines
        >>> number_of_pixels = [10, 10]
        >>> pixels = pept.Pixels.from_lines(lines2d, number_of_pixels)
        >>> grapher.add_lines(lines)
        >>> grapher.add_trace(pixels.pixels_trace())
        >>> grapher.show()

        '''

        filtered = self.copy()
        filtered[~condition(self)] = 0.

        # Compute the pixel centres
        x = self.pixel_grids[0]
        x = (x[1:] + x[:-1]) / 2

        y = self.pixel_grids[1]
        y = (y[1:] + y[:-1]) / 2

        trace = go.Surface(
            x = x,
            y = y,
            z = filtered,
            opacity = opacity,
            colorscale = colorscale,
        )

        return trace


    def heatmap_trace(
        self,
        colorscale = "Magma",
        transpose = True,
        xgap = 0.,
        ygap = 0.,
    ):
        '''Create and return a Plotly `Heatmap` trace of the pixels.

        Parameters
        ----------
        colorscale : str, default "Magma"
            The Plotly scheme for color-coding the pixel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        transpose : bool, default True
            Transpose the heatmap (i.e. flip it across its diagonal).

        Examples
        --------
        Pixellise an array of lines and add them to a `PlotlyGrapher2D`
        instance:

        >>> lines = np.array(...)                   # shape (N, M >= 7)
        >>> lines2d = lines[:, [0, 1, 2, 4, 5]]     # select x, y of lines
        >>> number_of_pixels = [10, 10]
        >>> pixels = pept.Pixels.from_lines(lines2d, number_of_pixels)

        >>> grapher = pept.visualisation.PlotlyGrapher2D()
        >>> grapher.add_pixels(pixels)
        >>> grapher.show()

        Or add them directly to a raw `plotly.graph_objs` figure:

        >>> import plotly.graph_objs as go
        >>> fig = go.Figure()
        >>> fig.add_trace(pixels.heatmap_trace())
        >>> fig.show()

        '''

        # Compute the pixel centres
        x = self.pixel_grids[0]
        x = (x[1:] + x[:-1]) / 2

        y = self.pixel_grids[1]
        y = (y[1:] + y[:-1]) / 2

        heatmap = dict(
            x = x,
            y = y,
            z = self,
            colorscale = colorscale,
            transpose = transpose,
            xgap = xgap,
            ygap = ygap,
        )

        return go.Heatmap(heatmap)


    def plot(self, ax = None):
        '''Plot pixels as a heatmap using Matplotlib.

        Returns matplotlib figure and axes objects containing the pixel values
        colour-coded in a Matplotlib image (i.e. heatmap).

        Parameters
        ----------
        ax : mpl_toolkits.mplot3D.Axes3D object, optional
            The 3D matplotlib-based axis for plotting. If undefined, new
            Matplotlib figure and axis objects are created.

        Returns
        -------
        fig, ax : matplotlib figure and axes objects

        Examples
        --------
        Pixellise an array of lines and plot them with Matplotlib:

        >>> lines = np.array(...)                   # shape (N, M >= 7)
        >>> lines2d = lines[:, [0, 1, 2, 4, 5]]     # select x, y of lines
        >>> number_of_pixels = [10, 10]
        >>> pixels = pept.Pixels.from_lines(lines2d, number_of_pixels)

        >>> fig, ax = pixels.plot()
        >>> fig.show()

        '''

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = plt.gcf()

        # Plot the values in pixels (this class is a numpy array subclass)
        ax.imshow(self)

        # Compute the pixel centres and set them in the Matplotlib image
        x = self.pixel_grids[0]
        x = (x[1:] + x[:-1]) / 2

        y = self.pixel_grids[1]
        y = (y[1:] + y[:-1]) / 2

        # Matplotlib shows numbers in a long format ("102.000032411"), so round
        # them to two decimals before plotting
        ax.set_xticklabels(np.round(x, 2))
        ax.set_yticklabels(np.round(y, 2))

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        return fig, ax


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"{self.__array__()}\n\n"
            f"number_of_pixels =    {self._number_of_pixels}\n"
            f"pixel_size =          {self._pixel_size}\n\n"
            f"xlim =                {self._xlim}\n"
            f"ylim =                {self._ylim}\n"
            f"pixel_grids:\n"
            f"([{self._pixel_grids[0][0]} ... {self._pixel_grids[0][-1]}],\n"
            f" [{self._pixel_grids[1][0]} ... {self._pixel_grids[1][-1]}])"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPR
        docstr = (
            "Class instance that inherits from `pept.Pixels`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n----------\n"
            f"{self.__str__()}"
        )

        return docstr
