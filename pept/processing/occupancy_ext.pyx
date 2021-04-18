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
#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools


# File              : occupancy_ext.pyx
# License           : GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 06.04.2020


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


#import numpy as np

from libc.stdint cimport int32_t


cdef extern from "rtd2d.c":
    # C is included here so that it doesn't need to be compiled externally
    pass


cdef extern from "rtd2d.h":
    void rtd2d_occupancy(
        float           *pixels,
        const int32_t   dims[2],
        const float     xmax[2],
        const float     ymax[2],
        const float     *positions,
        const float     *times,
        const int32_t   num_positions,
        const float     radius,
        const int32_t   omit_first
    ) nogil


cpdef void occupancy2d_ext(
    float[:, :] pixels,
    const float[:] xlim,
    const float[:] ylim,
    const float[:, :] positions,
    const float[:] times,
    const float radius,
    bint omit_last = False,
) nogil:
    '''Compute the 2D occupancy of a single circular particle moving along a
    trajectory.

    ::

        Function signature:

            void occupancy2d_ext(
                float[:, :] pixels,
                float[:] xlim,
                float[:] ylim,
                float[:, :] positions,
                float[:] times,
                float radius,
                bool omit_last = False,
            ) nogil

    This corresponds to the pixellisation of moving circular particles, such
    that for every two consecutive particle locations, a 2D cylinder (i.e.
    convex hull of two circles at the two particle positions), the fraction of
    its area that intersets a pixel is multiplied with the time between the
    two particle locations and saved in the input `pixels`.

    All parameters sent to this function should be initialised properly. Please
    see the parameters' description below, or the function *will* segfault.
    Alternatively, use the `pept.processing.occupancy2d` function which is
    more robust and initialises all parameters for you.

    Parameters
    ----------
    pixels: (M, N) numpy.ndarray[ndim = 2, dtype = numpy.float32]
        The 2D grid of pixels, initialised to zero. It can be created with
        `numpy.zeros((nrows, ncols))`. Important that it contains
        single-precision values.

    xlim: np.float32
        The upper physical limit of the system spanned by `pixels` in the
        x-dimension. Important: the lower limit is always assumed to be at
        (0, 0). You can translate the particle positions before calling this
        function (`pept.processing.occupancy2d does this automatically`).

    ylim: np.float32
        The upper physical limit of the system spanned by `pixels` in the
        y-dimension. Important: the lower limit is always assumed to be at
        (0, 0). You can translate the particle positions before calling this
        function (`pept.processing.occupancy2d does this automatically`).

    positions: (P, 2) numpy.ndarray[ndim = 2, dtype = numpy.float32]
        The particles' 2D positions, where all rows are formatted as
        `[x_coordinate, y_coordinate]`.

    times: (P,) numpy.ndarray[ndim = 1, dtype = numpy.float32]
        The corresponding timestamp for each particle location in `positions`.

    radius: numpy.float32
        The particle's radius. Can be in any system of units, as long as it is
        consistent with what is used for the particle `positions`.

    omit_last: bool, default False
        If True, omit the last circle in the particle positions. Useful if
        rasterizing the same trajectory piece-wise; if you split the trajectory
        and call this function multiple times, set `omit_last = 0` to avoid
        considering the last particle location twice.

    '''

    cdef int32_t    dims[2]
    cdef int32_t    num_positions
    cdef int32_t    omit_last_i32 = omit_last

    dims[0] = pixels.shape[0]
    dims[1] = pixels.shape[1]

    num_positions = positions.shape[0]

    rtd2d_occupancy(
        &pixels[0, 0],
        dims,
        &xlim[0],
        &ylim[0],
        &positions[0, 0],
        &times[0],
        num_positions,
        radius,
        omit_last_i32
    )
