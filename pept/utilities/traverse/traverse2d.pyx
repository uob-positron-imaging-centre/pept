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


# File              : traverse2d.pyx
# License           : GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 03.02.2019


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


from libc.float cimport DBL_MAX


cdef inline double fabs(double x) nogil:
    return (x if x >= 0 else -x)


cdef inline void swap(double *x, double *y) nogil:
    cdef double aux

    aux = x[0]
    x[0] = y[0]
    y[0] = aux


cdef double intersect(
    double[2] u,
    double[2] v,
    double xmin,
    double xmax,
    double ymin,
    double ymax,
) nogil:
    '''Given a 2D line defined as L(t) = U + t V and an axis-aligned bounding
    box (AABB), find the `t` for which the line intersects the box, if it
    exists.

    It is assumed that the line starts *outside* the AABB, so that t == 0.0 is
    only reserved for when the line does not intersect the box.
    '''

    cdef double tmin, tmax, tymin, tymax

    tmin = (xmin - u[0]) / v[0]     # Relies on IEEE FP behaviour for div by 0
    tmax = (xmax - u[0]) / v[0]

    tymin = (ymin - u[1]) / v[1]
    tymax = (ymax - u[1]) / v[1]

    if tmin > tmax: swap(&tmin, &tmax)
    if tymin > tymax: swap(&tymin, &tymax)

    if tmin > tymax or tymin > tmax: return 0.0

    if tymin > tmin: tmin = tymin
    if tymax < tmax: tmax = tymax

    return tmin


cpdef void traverse2d(
    double[:, :] pixels,              # Initialised to zero!
    const double[:, :] lines,         # Has exactly 5 columns!
    const double[:] grid_x,           # Has pixels.shape[0] + 1 elements!
    const double[:] grid_y,           # Has pixels.shape[1] + 1 elements!
) nogil:
    ''' Fast pixel traversal for 2D lines (or LoRs).

    ::

        Function Signature:
            traverse2d(
                double[:, :] pixels,        # Initialised to zero!
                double[:, :] lines,         # Has exactly 7 columns!
                double[:] grid_x,           # Has pixels.shape[0] + 1 elements!
                double[:] grid_y,           # Has pixels.shape[1] + 1 elements!
            )

    This function computes the number of lines that passes through each pixel,
    saving the result in `pixels`. It does so in an efficient manner, in which
    for every line, only the pixels that it passes through are traversed.

    As it is highly optimised, this function does not perform any checks on the
    validity of the input data. Please check the parameters before calling
    `traverse2d`, as it WILL segfault on wrong input data. Details are given
    below, along with an example call.

    Parameters
    ----------
    pixels : numpy.ndarray(dtype = numpy.float64, ndim = 2)
        The `pixels` parameter is a numpy.ndarray of shape (X, Y) that has been
        initialised to zeros before the function call. The values will be
        modified in-place in the function to reflect the number of lines that
        pass through each pixel.

    lines : numpy.ndarray(dtype = numpy.float64, ndim = 2)
        The `lines` parameter is a numpy.ndarray of shape(N, 5), where each row
        is formatted as [time, x1, y1, x2, y2]. Only indices 1:5 will be used
        as the two points P1 = [x1, y1] and P2 = [x2, y2] defining the line (or
        LoR).

    grid_x : numpy.ndarray(dtype = numpy.float64, ndim = 1)
        The grid_x parameter is a one-dimensional grid that delimits the pixels
        in the x-dimension. It must be *sorted* in ascending order with
        *equally-spaced* numbers and length X + 1 (pixels.shape[0] + 1).

    grid_y : numpy.ndarray(dtype = numpy.float64, ndim = 1)
        The grid_y parameter is a one-dimensional grid that delimits the pixels
        in the y-dimension. It must be *sorted* in ascending order with
        *equally-spaced* numbers and length Y + 1 (pixels.shape[1] + 1).

    Examples
    --------
    The input parameters can be easily generated using numpy before calling the
    function. For example, if a plane of 300 x 400 is split into
    30 x 40 pixels, a possible code would be:

    >>> import numpy as np
    >>> from pept.utilities.traverse import traverse2d
    >>>
    >>> plane = [300, 400]
    >>> number_of_pixels = [30, 40]
    >>> pixels = np.zeros(number_of_pixels)

    The grid has one extra element than the number of pixels. For example, 5
    pixels between 0 and 5 would be delimited by the grid [0, 1, 2, 3, 4, 5]
    which has 6 elements (see off-by-one errors - story of my life).

    >>> grid_x = np.linspace(0, plane[0], number_of_pixels[0] + 1)
    >>> grid_y = np.linspace(0, plane[1], number_of_pixels[1] + 1)
    >>>
    >>> random_lines = np.random.random((100, 5)) * 100

    Calling `traverse2d` will modify `pixels` in-place.

    >>> traverse2d(pixels, random_lines, grid_x, grid_y)

    Notes
    -----
    This function is an adaptation of a widely-used algorithm [1]_, optimised
    for PEPT LoRs traversal.

    .. [1] Amanatides J, Woo A. A fast voxel traversal algorithm for ray tracing.
       InEurographics 1987 Aug 24 (Vol. 87, No. 3, pp. 3-10).

    '''

    n_lines = lines.shape[0]
    cdef Py_ssize_t nx = pixels.shape[0]
    cdef Py_ssize_t ny = pixels.shape[1]

    # Grid size
    cdef double gsize_x = grid_x[1] - grid_x[0]
    cdef double gsize_y = grid_y[1] - grid_y[0]

    # Delimiting grid
    cdef double xmin = grid_x[0]
    cdef double xmax = grid_x[nx]

    cdef double ymin = grid_y[0]
    cdef double ymax = grid_y[ny]

    # The current pixel indices [ix, iy] that the line passes
    # through.
    cdef Py_ssize_t ix, iy

    # Define a line as L(t) = U + t V
    # If an LoR is defined as two points P1 and P2, then
    # U = P1 and V = P2 - P1
    cdef double[2] p1, p2, u, v

    # The step [step_x, step_y, step_z] defines the sense of the LoR.
    # If V[0] is positive, then step_x = 1
    # If V[0] is negative, then step_x = -1
    cdef Py_ssize_t step_x, step_y

    # The value of t at which the line passes through to the next
    # pixel, for each dimension.
    cdef double tnext_x, tnext_y

    # deltat indicates how far along the ray we must move (in units of
    # t) for each component to be equal to the size of the pixel in
    # that dimension.
    cdef double deltat_x, deltat_y

    cdef Py_ssize_t i, j

    for i in range(n_lines):

        for j in range(2):
            p1[j] = lines[i, 1 + j]
            p2[j] = lines[i, 3 + j]

            u[j] = p1[j]
            v[j] = p2[j] - p1[j]

        ##############################################################
        # Initialisation stage

        step_x = 1 if v[0] >= 0 else -1
        step_y = 1 if v[1] >= 0 else -1

        # If the first point is outside the box, find the first pixel it hits
        if (u[0] < xmin or u[0] > xmax or
                u[1] < ymin or u[1] > ymax):

            t = intersect(u, v, xmin, xmax, ymin, ymax)

            # No intersection
            if t == 0.0:
                continue

            # Overwrite U to correspond to the first intersection point
            for j in range(2):
                u[j] = u[j] + t * v[j]

        # Corner case: every pixel is defined as lower boundary (inclusive) and
        # upper boundary (exclusive). Therefore, at the upper end of the pixel
        # grid an undefined case occurs. If a point lies right at the upper
        # boundary of the pixel space, "move it" a bit lower on the line
        if u[0] == xmax or u[1] == ymax:
            for j in range(2):
                u[j] = u[j] + 1e-5 * v[j]

        # If, for dimension x, there are 5 pixels between coordinates 0
        # and 5, then the delimiting grid is [0, 1, 2, 3, 4, 5].
        # If the line starts at 1.5, then it is part of the pixel at
        # index 1

        ix = <int>(u[0] / gsize_x)
        iy = <int>(u[1] / gsize_y)

        # Check the indices are inside the pixel grid
        if (ix < 0 or ix >= nx or iy < 0 or iy >= ny):
            continue

        # If the line is going "up", the next pixel is the next one
        # If the line is going "down", the next pixel is the current one
        if v[0] > 0:
            tnext_x = (grid_x[ix + 1] - u[0]) / v[0]
        elif v[0] < 0:
            tnext_x = (grid_x[ix] - u[0]) / v[0]
        else:
            tnext_x = DBL_MAX

        if v[1] > 0:
            tnext_y = (grid_y[iy + 1] - u[1]) / v[1]
        elif v[1] < 0:
            tnext_y = (grid_y[iy] - u[1]) / v[1]
        else:
            tnext_y = DBL_MAX

        deltat_x = fabs((grid_x[1] - grid_x[0]) / v[0]) if v[0] else 0
        deltat_y = fabs((grid_y[1] - grid_y[0]) / v[1]) if v[1] else 0

        ###############################################################
        # Incremental traversal stage

        # Loop until we reach the last pixel in space
        while (ix < nx and iy < ny) and (ix >= 0 and iy >= 0):

            pixels[ix, iy] += 1

            # Select the minimum t that makes the line pass
            # through to the next pixel
            if tnext_x < tnext_y:
                # If the next pixel falls beyond the end of the line (that is
                # at t = 1), then stop the traversal stage
                if tnext_x > 1.:
                    break

                ix = ix + step_x
                tnext_x = tnext_x + deltat_x
            else:
                if tnext_y > 1.:
                    break

                iy = iy + step_y
                tnext_y = tnext_y + deltat_y

