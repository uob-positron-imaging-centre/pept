#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : occupancy.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 23.11.2020


import  time
import  textwrap

import  numpy           as      np
import  pept

from    .occupancy_ext  import  occupancy2d_ext


def occupancy2d(
    positions,
    number_of_pixels,
    radii = 0.,
    xlim = None,
    ylim = None,
    verbose = True,
):
    '''Compute the 2D occupancy of circles of different radii.

    This corresponds to the pixellisation of circular particles, such that
    each pixel's value corresponds to the area covered by the particle.

    You must specify the particles' `positions` (2D numpy array) and the
    `number_of_pixels` in each dimension (`[nx, ny, nz]`). The `radii` can be
    either:

    1. Zero: the particles are considered to be points. Each pixel will have a
       value +1 for every particle.
    2. Single positive value: all particles have the same radius.
    3. List of values of same length as `positions`: specify each particle's
       radius.

    The pixel area's bounds can be specified in `xlim` and `ylim`. If unset,
    they will be computed automatically based on the minimum and maximum
    values found in `positions`.

    Parameters
    ----------
    pixels: (M, N) numpy.ndarray[ndim = 2, dtype = numpy.float64]
        The 2D grid of pixels, initialised to zero. It can be created with
        `numpy.zeros((nrows, ncols))`.

    positions: (P, 2) numpy.ndarray
        The particles' 2D positions, where each row is formatted as
        `[x_coordinate, y_coordinate]`.

    number_of_pixels: (2,) list-like
        The number of pixels in the x-dimension and y-dimension. Each dimension
        must have at least 2 pixels.

    radii: float or (P,) list-like
        The radius of each particle. If zero, every particle is considered as
        a discrete point. If a single `float`, all particles are considered to
        have the same radius. If it is a numpy array, it specifies each
        particle's radius, and must have the same length as `positions`.

    xlim: (2,) list-like, optional
        The limits of the system over which the pixels span in the
        x-dimension, formatted as [xmin, xmax]. If unset, they will be computed
        automatically based on the minimum and maximum values found in
        `positions`.

    ylim: (2,) list-like, optional
        The limits of the system over which the pixels span in the
        y-dimension, formatted as [ymin, ymax]. If unset, they will be computed
        automatically based on the minimum and maximum values found in
        `positions`.

    verbose: bool, default True
        Time the pixellisation step and print it to the terminal.

    Returns
    -------
    pept.Pixels (numpy.ndarray subclass)
        The created pixels, each cell containing the area covered by particles.
        The `pept.Pixels` class inherits all properties and methods from
        `numpy.ndarray`, so you can use it exactly like you would a numpy
        array. It just contains extra attributes (e.g. `xlim`, `ylim`) and
        some PEPT-oriented methods (e.g. `pixels_trace`).

    Raises
    ------
    ValueError
        If `positions` is not a 2D array-like with exactly 2 columns, or
        `number_of_pixels` is not a 1D list-like with exactly 2 values or it
        contains a value smaller than 2. If `radii` is a list-like that does
        not have the same length as `positions`.

    Examples
    --------
    Create ten random particle positions between 0-100 and radii between
    0.5-2.5:
    >>> positions = np.random.random((10, 2)) * 100
    >>> radii = 0.5 + np.random.random(len(positions)) * 2

    Now pixellise those particles over a grid of (20, 10) pixels:
    >>> import pept.processing as pp
    >>> num_pixels = (20, 10)
    >>> pixels = pp.occupancy2d(positions, num_pixels, radii)

    Alternatively, specify the system's bounds explicitly:
    >>> pixels = pp.occupancy2d(
    >>>     positions, (20, 10), radii, xlim = [10, 90], ylim = [-5, 105]
    >>> )

    You can plot those pixels in two ways - using `PlotlyGrapher` (this plots a
    3D "heatmap", as a coloured surface):
    >>> from pept.visualisation import PlotlyGrapher
    >>> grapher = PlotlyGrapher()
    >>> grapher.add_pixels(pixels)
    >>> grapher.show()

    Or using raw `Plotly` (this plots a "true" heatmap):
    >>> import plotly.graph_objs as go
    >>> fig = go.Figure()
    >>> fig.add_trace(pixels.heatmap_trace())
    >>> fig.show()

    '''
    if verbose:
        start = time.time()

    # Type-checking inputs
    positions = np.asarray(positions, order = 'C', dtype = float)

    # Check that points has at least 4 columns.
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(textwrap.fill((
            "The input `positions` should have exactly 2 columns, "
            "corresponding to the [x, y] coordinates of the particle "
            f"positions. Received array with shape {positions.shape}."
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

    # `radii` is either a float, or a numpy array
    try:
        radii = float(radii)
        radii = np.ones(len(positions)) * radii
    except Exception:
        radii = np.asarray(radii, order = 'C', dtype = float)
        if radii.ndim != 1 or len(radii) != len(positions):
            raise ValueError(textwrap.fill((
                "The input `radii` must be a float (i.e. single radius "
                "for all particles) or a 1D numpy array of radii for each "
                "particle in the system - so `radii` must have the same "
                "length as `positions`. Received array with length "
                f"{len(radii)} != {len(positions)}."
            )))

    if xlim is None:
        xlim = np.array([
            positions[:, 0].min() - radii.max(),
            positions[:, 0].max() + radii.max(),
        ])
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
        ylim = np.array([
            positions[:, 1].min() - radii.max(),
            positions[:, 1].max() + radii.max(),
        ])
    else:
        ylim = np.asarray(ylim, dtype = float)

        if ylim.ndim != 1 or len(ylim) != 2:
            raise ValueError(textwrap.fill((
                "The input `ylim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the y-dimension. "
                f"Received parameter with shape {ylim.shape}."
            )))

    pixels = np.zeros(tuple(number_of_pixels), order = "C", dtype = float)

    # This modifies `pixels` in-place
    occupancy2d_ext(
        pixels,
        positions,
        radii,
        xlim,
        ylim,
    )

    occupancy_grid = pept.Pixels(pixels, xlim = xlim, ylim = ylim)

    if verbose:
        end = time.time()
        print(f"Computed occupancy grid in {end - start} s")

    return occupancy_grid














