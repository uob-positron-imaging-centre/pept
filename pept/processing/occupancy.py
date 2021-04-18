#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools.
#
#    If you used this codebase or any software making use of it in a scientific
#    publication, you should cite the following paper:
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


# File   : occupancy.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 23.11.2020




import  time
import  textwrap

import  numpy           as      np
import  pept

from    .circles_ext    import  circles2d_ext
from    .occupancy_ext  import  occupancy2d_ext


def circles2d(
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

    Now pixellise those particles as circles over a grid of (20, 10) pixels:

    >>> import pept.processing as pp
    >>> num_pixels = (20, 10)
    >>> pixels = pp.circles2d(positions, num_pixels, radii)

    Alternatively, specify the system's bounds explicitly:

    >>> pixels = pp.circles2d(
    >>>     positions, (20, 10), radii, xlim = [10, 90], ylim = [-5, 105]
    >>> )

    You can plot those pixels in two ways - using `PlotlyGrapher` (this plots a
    3D "heatmap", as a coloured surface):

    >>> from pept.visualisation import PlotlyGrapher
    >>> grapher = PlotlyGrapher()
    >>> grapher.add_pixels(pixels)
    >>> grapher.show()

    Or using raw `Plotly` (this plots a "true" heatmap) - this is recommended:

    >>> import plotly.graph_objs as go
    >>> fig = go.Figure()
    >>> fig.add_trace(pixels.heatmap_trace())
    >>> fig.show()

    '''
    if verbose:
        start = time.time()

    # Type-checking inputs
    positions = np.asarray(positions, order = 'C', dtype = np.float64)

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
        radii = np.asarray(radii, order = 'C', dtype = np.float64)
        if radii.ndim != 1 or len(radii) != len(positions):
            raise ValueError(textwrap.fill((
                "The input `radii` must be a float (i.e. single radius "
                "for all particles) or a 1D numpy array of radii for each "
                "particle in the system - so `radii` must have the same "
                "length as `positions`. Received array with length "
                f"{len(radii)} != {len(positions)}."
            )))

    # Maximum particle radius
    max_radius = radii.max()

    if xlim is None:
        xmin = (positions[:, 0].min() - max_radius)
        xmax = (positions[:, 0].max() + max_radius)

        xlim = np.array([
            xmin - 0.01 * abs(xmin),
            xmax + 0.01 * abs(xmax),
        ])
    else:
        xlim = np.asarray(xlim, dtype = np.float64)

        if xlim.ndim != 1 or len(xlim) != 2 or xlim[0] >= xlim[1]:
            raise ValueError(textwrap.fill((
                "The input `xlim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the x-dimension. "
                f"Received parameter with shape {xlim.shape}."
            )))

    if ylim is None:
        ymin = (positions[:, 1].min() - max_radius)
        ymax = (positions[:, 1].max() + max_radius)

        ylim = np.array([
            ymin - 0.01 * abs(ymin),
            ymax + 0.01 * abs(ymax),
        ])
    else:
        ylim = np.asarray(ylim, dtype = np.float64)

        if ylim.ndim != 1 or len(ylim) != 2 or ylim[0] >= ylim[1]:
            raise ValueError(textwrap.fill((
                "The input `ylim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the y-dimension. "
                f"Received parameter with shape {ylim.shape}."
            )))

    pixels = np.zeros(tuple(number_of_pixels), order = "C", dtype = np.float64)

    # This modifies `pixels` in-place
    circles2d_ext(
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


def occupancy2d(
    points,
    number_of_pixels,
    radius,
    xlim = None,
    ylim = None,
    omit_last = False,
    verbose = True,
):
    '''Compute the 2D occupancy / residence time distribution of a single
    circular particle moving along a trajectory.

    This corresponds to the pixellisation of moving circular particles, such
    that for every two consecutive particle locations, a 2D cylinder (i.e.
    convex hull of two circles at the two particle positions), the fraction of
    its area that intersets a pixel is multiplied with the time between the
    two particle locations and saved in the input `pixels`.

    You must specify the `points` (2D numpy array) recorded along a particle's
    trajectory, formatted as [time, x, y] of each location, along with the
    `number_of_pixels` in each dimension (`[nx, ny]`) and particle `radius`.

    The pixel area's bounds can be specified in `xlim` and `ylim`. If unset,
    they will be computed automatically based on the minimum and maximum
    values found in `points`.

    Parameters
    ----------
    points: (P, 3) numpy.ndarray
        The particles' 2D locations and corresponding timestamp, where each row
        is formatted as `[time, x_coordinate, y_coordinate]`. Must have at
        least two points.

    number_of_pixels: (2,) list-like
        The number of pixels in the x-dimension and y-dimension. Each dimension
        must have at least 2 pixels.

    radius: float
        The radius of the particle. It can be given in any system of units, as
        long as it is consistent with what is used for the particle locations.

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

    omit_last: bool, default False
        If true, omit the last circle in the particle positions. Useful if
        rasterizing the same trajectory piece-wise; if you split the trajectory
        and call this function multiple times, set `omit_last = 0` to avoid
        considering the last particle location twice.

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
        If `positions` is not a 2D array-like with exactly 3 columns, or
        `number_of_pixels` is not a 1D list-like with exactly 2 values or it
        contains a value smaller than 2. If `xlim` or `ylim` have `max` < `min`
        or there are particle positions falling outside the system defined by
        `xlim` and `ylim`, including the area.

    Examples
    --------
    Create ten random particle positions between 0-100 and radius 0.2:

    >>> positions = np.random.random((10, 2)) * 100
    >>> radius = 0.2

    Now pixellise this trajectory over a grid of (20, 10) pixels:

    >>> import pept.processing as pp
    >>> num_pixels = (20, 10)
    >>> pixels = pp.occupancy2d(positions, num_pixels, radius)

    Alternatively, specify the system's bounds explicitly:

    >>> pixels = pp.occupancy2d(
    >>>     positions, (20, 10), radius, xlim = [10, 90], ylim = [-5, 105]
    >>> )

    You can plot those pixels in two ways - using `PlotlyGrapher` (this plots a
    3D "heatmap", as a coloured surface):

    >>> from pept.visualisation import PlotlyGrapher
    >>> grapher = PlotlyGrapher()
    >>> grapher.add_pixels(pixels)
    >>> grapher.show()

    Or using raw `Plotly` (this plots a "true" heatmap) - this is recommended:

    >>> import plotly.graph_objs as go
    >>> fig = go.Figure()
    >>> fig.add_trace(pixels.heatmap_trace())
    >>> fig.show()

    '''
    if verbose:
        start = time.time()

    # Type-checking inputs
    points = np.array(points, order = 'C', dtype = np.float32)

    # Check that points has exactly 3 columns for [time, x, y]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(textwrap.fill((
            "The input `points` should have exactly 3 columns, "
            "formatted as the [time, x, y] coordinates of the particle "
            f"positions. Received array with shape {points.shape}."
        )))

    times = np.array(points[:, 0], order = "C", dtype = np.float32)
    positions = np.array(points[:, 1:], order = "C", dtype = np.float32)

    number_of_pixels = np.asarray(
        number_of_pixels,
        order = "C",
        dtype = np.int32,
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

    radius = np.float32(radius)

    if xlim is None:
        xlim = np.array([
            positions[:, 0].min() - radius,
            positions[:, 0].max() + radius,
        ], dtype = np.float32)
    else:
        xlim = np.asarray(xlim, dtype = np.float32)

        if xlim.ndim != 1 or len(xlim) != 2 or xlim[0] >= xlim[1]:
            raise ValueError(textwrap.fill((
                "The input `xlim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the x-dimension. "
                f"Received parameter with shape {xlim.shape}."
            )))

    if ylim is None:
        ylim = np.array([
            positions[:, 1].min() - radius,
            positions[:, 1].max() + radius,
        ], dtype = np.float32)
    else:
        ylim = np.asarray(ylim, dtype = np.float32)

        if ylim.ndim != 1 or len(ylim) != 2 or ylim[0] >= ylim[1]:
            raise ValueError(textwrap.fill((
                "The input `ylim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the y-dimension. "
                f"Received parameter with shape {ylim.shape}."
            )))

    omit_last = bool(omit_last)

    # `occupancy2d_ext` expects a single-precision pixels array
    grid = np.zeros(tuple(number_of_pixels), order = "C", dtype = np.float32)

    # This modifies `pixels` in-place
    occupancy2d_ext(
        grid,
        xlim,
        ylim,
        positions,
        times,
        radius,
        omit_last,
    )

    pixels = pept.Pixels(grid, xlim = xlim, ylim = ylim)

    if verbose:
        end = time.time()
        print(f"Computed occupancy grid in {end - start} s")

    return pixels
