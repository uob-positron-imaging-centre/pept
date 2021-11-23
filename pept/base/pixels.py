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


# File   : pixels.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 23.11.2021


import  time
import  textwrap

import  numpy                   as      np

from    konigcell               import  Pixels
from    pept.utilities.traverse import  traverse2d



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
        xlim = get_cutoff(lines[:, 1], lines[:, 3])
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
        ylim = get_cutoff(lines[:, 1], lines[:, 3])
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


# Dynamically add this as a Pixels method
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
        self.pixel_grids[0],
        self.pixel_grids[1],
    )

    if verbose:
        end = time.time()
        print(f"Traversing {len(lines)} lines took {end - start} s.")


# Add the `from_lines` function as a static method to Pixels
Pixels.from_lines = staticmethod(from_lines)
Pixels.add_lines = add_lines
