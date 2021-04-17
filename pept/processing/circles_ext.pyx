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
# Date              : 23.11.2020


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


#import numpy as np

from libc.math cimport floor, sqrt, acos, M_PI
from libc.float cimport DBL_MAX


cdef inline double fabs(double x) nogil:
    return (x if x >= 0 else -x)


cdef inline double dist(double x1, double y1, double x2, double y2) nogil:
    # Distance between two 2D points
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


cdef inline double dist2(double x1, double y1, double x2, double y2) nogil:
    # Squared distance between two 2D points
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


cdef inline double term(double r, double _c, double _p) nogil:
    # Term in calculation of `one_corner`
    # Same for xc / xp and yc / yp
    return sqrt(r ** 2 - (_c - _p) ** 2)


cdef inline double triangle(
    double x1, double y1, double x2, double y2, double x3, double y3
) nogil:
    return 0.5 * fabs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


cdef inline double dist_pl(
    double x0,
    double y0,
    double x1,
    double y1,
    double x2,
    double y2
) nogil:
    # Distance from a point (x0, y0) to a line defined by two points
    # (x1, y1) and (x2, y2)
    denom = dist(x1, y1, x2, y2)
    nom = fabs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)

    return nom / denom


cdef inline double circular_segment(double r, double h) nogil:
    # Area of a circular segment from a circle of radius r and a line
    # If the distance from the circle centre to the line is d, then
    # h = r - d
    if h < 0.0:     # Happens for very small distances (d = r * 1e-13)
        return 0.0

    return r ** 2 * acos((r - h) / r) - (r - h) * sqrt(2 * r * h - h ** 2)


cdef inline double zero_corner(
    Py_ssize_t[4] corners,
    double px,
    double py,
    double xsize,
    double ysize,
    double x0,
    double y0,
    double r,
) nogil:
    '''Compute the overlapping area between a circle and a rectangle when
    exactly one of the rectangle's corners are inside the circle.

    Parameters
    ----------
    corners
        Vector of 0 and 1 for the corner that is inside the circle. The
        corners are in trigonometric order.

    px, py
        The pixel x and y coordinates.

    xsize, ysize
        The pixel width and height.

    x, y
        The circle origin x and y coordinates

    r
        The circle radius

    Returns
    -------
    The overlapping area.
    '''

    # Distance from circle centre x0, y0 to line defined by the two
    # intersections above and h = (r - d)
    cdef double h
    cdef double area = 0.
    cdef double xmin = px - xsize / 2
    cdef double xmax = px + xsize / 2
    cdef double ymin = py - ysize / 2
    cdef double ymax = py + ysize / 2

    # Corner case - if zero corners are inside, there might be no intersection
    # This happens when the centre of the particle is in one of the corners
    if ((x0 <= xmin and y0 <= ymin) or      # Bottom left
            (x0 >= xmax and y0 <= ymin) or  # Bottom right
            (x0 >= xmax and y0 >= ymax) or  # Top right
            (x0 <= xmin and y0 >= ymax)):   # Top left
        return 0.

    # Particle is fully encapsulated
    if xmin < x0 < xmax and ymin < y0 < ymax:
        return M_PI * r * r
    # Below
    elif xmin < x0 < xmax:
        h = ysize / 2 + r - fabs(py - y0)
    # Above
    else:
        h = xsize / 2 + r - fabs(px - x0)

    # Area of the circular segment
    area += circular_segment(r, h)

    return area


cdef inline double one_corner(
    Py_ssize_t[4] corners,
    double px,
    double py,
    double xsize,
    double ysize,
    double x0,
    double y0,
    double r,
) nogil:
    '''Compute the overlapping area between a circle and a rectangle when
    exactly one of the rectangle's corners are inside the circle.

    Parameters
    ----------
    corners
        Vector of 0 and 1 for the corner that is inside the circle. The
        corners are in trigonometric order.

    px, py
        The pixel x and y coordinates.

    xsize, ysize
        The pixel width and height.

    x, y
        The circle origin x and y coordinates

    r
        The circle radius

    Returns
    -------
    The overlapping area.
    '''

    # The bounded corner coordinates
    cdef double xc = 0, yc = 0

    # The coords of the two intersections between the circle and the rectangle
    cdef double x1 = 0, y1 = 0, x2 = 0, y2 = 0

    # Distance from circle centre x0, y0 to line defined by the two
    # intersections above and h = (r - d)
    cdef double d, h

    cdef double area = 0.

    # Pixel boundaries
    cdef double xmin = px - xsize / 2
    cdef double xmax = px + xsize / 2
    cdef double ymin = py - ysize / 2
    cdef double ymax = py + ysize / 2

    # Lower left corner
    if corners[0] == 1:
        xc = xmin
        yc = ymin

        x1 = x0 + term(r, yc, y0)
        y1 = yc

        x2 = xc
        y2 = y0 + term(r, xc, x0)

    # Lower right corner
    elif corners[1] == 1:
        xc = xmax
        yc = ymin

        x1 = xc
        y1 = y0 + term(r, xc, x0)

        x2 = x0 - term(r, yc, y0)
        y2 = yc

    # Upper right corner
    elif corners[2] == 1:
        xc = xmax
        yc = ymax

        x1 = x0 - term(r, yc, y0)
        y1 = yc

        x2 = xc
        y2 = y0 - term(r, xc, x0)

    # Upper left corner
    elif corners[3] == 1:
        xc = xmin
        yc = ymax

        x1 = x0 + term(r, yc, y0)
        y1 = yc

        x2 = xc
        y2 = y0 - term(r, xc, x0)

    d = dist_pl(x0, y0, x1, y1, x2, y2)
    h = r - d

    # Area of the circular segment
    area += circular_segment(r, h)

    # Area of the triangle
    area += dist(xc, yc, x1, y1) * dist(xc, yc, x2, y2) / 2

    return area


cdef inline double two_corner(
    Py_ssize_t[4] corners,
    double px,
    double py,
    double xsize,
    double ysize,
    double x0,
    double y0,
    double r,
) nogil:
    '''Compute the overlapping area between a circle and a rectangle when
    exactly one of the rectangle's corners are inside the circle.

    Parameters
    ----------
    corners
        Vector of 0 and 1 for the corner that is inside the circle. The
        corners are in trigonometric order.

    px, py
        The pixel x and y coordinates.

    xsize, ysize
        The pixel width and height.

    x, y
        The circle origin x and y coordinates

    r
        The circle radius

    Returns
    -------
    The overlapping area.
    '''

    # The bounded corner coordinates
    cdef double xc1 = 0, yc1 = 0, xc2 = 0, yc2 = 0

    # The coords of the two intersections between the circle and the rectangle
    cdef double x1 = 0, y1 = 0, x2 = 0, y2 = 0

    # Distance from circle centre x0, y0 to line defined by the two
    # intersections above and h = (r - d)
    cdef double d, h

    cdef double area = 0.

    # Pixel boundaries
    cdef double xmin = px - xsize / 2
    cdef double xmax = px + xsize / 2
    cdef double ymin = py - ysize / 2
    cdef double ymax = py + ysize / 2

    # Bottom corners
    if corners[0] == 1 and corners[1] == 1:
        xc1 = xmin
        yc1 = ymin

        xc2 = xmax
        yc2 = ymin

        x1 = xc1
        y1 = y0 + term(r, xc1, x0)

        x2 = xc2
        y2 = y0 + term(r, xc2, x0)

    # Top corners
    elif corners[2] == 1 and corners[3] == 1:
        xc1 = xmin
        yc1 = ymax

        xc2 = xmax
        yc2 = ymax

        x1 = xc1
        y1 = y0 - term(r, xc1, x0)

        x2 = xc2
        y2 = y0 - term(r, xc2, x0)

    # Right corners
    elif corners[1] == 1 and corners[2] == 1:
        xc1 = xmax
        yc1 = ymin

        xc2 = xmax
        yc2 = ymax

        x1 = x0 - term(r, yc1, y0)
        y1 = yc1

        x2 = x0 - term(r, yc2, y0)
        y2 = yc2

    # Left corners
    elif corners[0] == 1 and corners[3] == 1:
        xc1 = xmin
        yc1 = ymin

        xc2 = xmin
        yc2 = ymax

        x1 = x0 + term(r, yc1, y0)
        y1 = yc1

        x2 = x0 + term(r, yc2, y0)
        y2 = yc2

    d = dist_pl(x0, y0, x1, y1, x2, y2)
    h = r - d

    # Area of the circular segment
    area += circular_segment(r, h)

    # Area of triangle1 and triangle2
    area += triangle(x1, y1, xc2, yc2, xc1, yc1)
    area += triangle(x1, y1, xc2, yc2, x2, y2)

    return area


cdef inline double three_corner(
    Py_ssize_t[4] corners,
    double px,
    double py,
    double xsize,
    double ysize,
    double x0,
    double y0,
    double r,
) nogil:
    '''Compute the overlapping area between a circle and a rectangle when
    exactly one of the rectangle's corners are inside the circle.

    Parameters
    ----------
    corners
        Vector of 0 and 1 for the corner that is inside the circle. The
        corners are in trigonometric order.

    px, py
        The pixel x and y coordinates.

    xsize, ysize
        The pixel width and height.

    x, y
        The circle origin x and y coordinates

    r
        The circle radius

    Returns
    -------
    The overlapping area.
    '''

    # The bounded corner coordinates
    cdef double xc1 = 0, yc1 = 0, xc2 = 0, yc2 = 0
    cdef double xc3 = 0, yc3 = 0, xc4 = 0, yc4 = 0

    # The coords of the two intersections between the circle and the rectangle
    cdef double x1 = 0, y1 = 0, x2 = 0, y2 = 0

    # Distance from circle centre x0, y0 to line defined by the two
    # intersections above and h = (r - d)
    cdef double d, h

    cdef double area = 0.

    # Pixel boundaries
    cdef double xmin = px - xsize / 2
    cdef double xmax = px + xsize / 2
    cdef double ymin = py - ysize / 2
    cdef double ymax = py + ysize / 2

    # Bottom right three corners
    if corners[0] == 1 and corners[1] == 1 and corners[2] == 1:
        xc1 = xmin
        yc1 = ymin

        xc2 = xmax
        yc2 = ymin

        xc3 = xmax
        yc3 = ymax

        xc4 = xmin    # Corner outside
        yc4 = ymax    # Corner outside

        x1 = x0 - term(r, yc3, y0)
        y1 = yc3

        x2 = xc1
        y2 = y0 + term(r, xc1, x0)

    # Top right three corners
    elif corners[1] == 1 and corners[2] == 1 and corners[3] == 1:
        xc1 = xmax
        yc1 = ymin

        xc2 = xmax
        yc2 = ymax

        xc3 = xmin
        yc3 = ymax

        xc4 = xmin    # Corner outside
        yc4 = ymin    # Corner outside

        x1 = xc3
        y1 = y0 - term(r, xc3, x0)

        x2 = x0 - term(r, yc1, y0)
        y2 = yc1

    # Top left three corners
    elif corners[2] == 1 and corners[3] == 1 and corners[0] == 1:
        xc1 = xmax
        yc1 = ymax

        xc2 = xmin
        yc2 = ymax

        xc3 = xmin
        yc3 = ymin

        xc4 = xmax    # Corner outside
        yc4 = ymin    # Corner outside

        x1 = xc1
        y1 = y0 - term(r, xc1, x0)

        x2 = x0 + term(r, yc3, y0)
        y2 = yc3

    # Bottom left three corners
    elif corners[0] == 1 and corners[1] == 1 and corners[3] == 1:
        xc1 = xmin
        yc1 = ymax

        xc2 = xmin
        yc2 = ymin

        xc3 = xmax
        yc3 = ymin

        xc4 = xmax    # Corner outside
        yc4 = ymax    # Corner outside

        x1 = xc3
        y1 = y0 + term(r, xc3, x0)

        x2 = x0 + term(r, yc1, y0)
        y2 = yc1

    d = dist_pl(x0, y0, x1, y1, x2, y2)
    h = r - d

    # Area of the circular segment
    area += circular_segment(r, h)

    # Area of the rectangle
    area += xsize * ysize

    # Area of triangle
    area -= dist(x1, y1, xc4, yc4) * dist(x2, y2, xc4, yc4) / 2

    return area


cdef inline double four_corner(
    Py_ssize_t[4] corners,
    double px,
    double py,
    double xsize,
    double ysize,
    double x0,
    double y0,
    double r,
) nogil:
    '''Compute the overlapping area between a circle and a rectangle when
    exactly one of the rectangle's corners are inside the circle.

    Parameters
    ----------
    corners
        Vector of 0 and 1 for the corner that is inside the circle. The
        corners are in trigonometric order.

    px, py
        The pixel x and y coordinates.

    xsize, ysize
        The pixel width and height.

    x, y
        The circle origin x and y coordinates

    r
        The circle radius

    Returns
    -------
    The overlapping area.
    '''

    return xsize * ysize



cpdef void circles2d_ext(
    double[:, :] pixels,
    double[:, :] positions,
    double[:] radii,
    double[:] xlim,
    double[:] ylim,
) nogil:
    '''Compute the 2D occupancy of circles of different radii.

    ::

        Function signature:

            void circles2d_ext(
                double[:, :] pixels,
                double[:, :] positions,
                double[:] radii,
                double[:] xlim,
                double[:] ylim,
            ) nogil

    This corresponds to the pixellisation of circular particles, such that
    each pixel's value corresponds to the area covered by the particle.

    It is important that all parameters sent to this function are initialised
    properly. Please see the parameters below, or the function *will* segfault.
    Alternatively, use the `pept.processing.occupancy2d` function which is
    more robust and initialises all parameters for you.

    Parameters
    ----------
    pixels: (M, N) numpy.ndarray[ndim = 2, dtype = numpy.float64]
        The 2D grid of pixels, initialised to zero. It can be created with
        `numpy.zeros((nrows, ncols))`.

    positions: (P, 2) numpy.ndarray[ndim = 2, dtype = numpy.float64]
        The particles' 2D positions, where all rows are formatted as
        `[x_coordinate, y_coordinate]`.

    radii: (P,) numpy.ndarray[ndim = 1, dtype = numpy.float64]
        The radii of each particle. It must have the same length as
        `positions`.

    xlim: (2,) numpy.ndarray[ndim = 1, dtype = numpy.float64]
        The limits of the system over which the pixels span in the
        x-dimension, formatted as [xmin, xmax].

    ylim: (2,) numpy.ndarray[ndim = 1, dtype = numpy.float64]
        The limits of the system over which the pixels span in the
        y-dimension, formatted as [ymin, ymax].

    '''
    cdef Py_ssize_t     nrows = positions.shape[0]

    cdef Py_ssize_t     nx = pixels.shape[0]
    cdef Py_ssize_t     ny = pixels.shape[1]

    cdef double         xsize = (xlim[1] - xlim[0]) / nx
    cdef double         ysize = (ylim[1] - ylim[0]) / ny

    # print((
    #     f"nrows = {nrows}\n"
    #     f"nx = {nx}\n"
    #     f"ny = {ny}\n"
    #     f"xsize = {xsize}\n"
    #     f"ysize = {ysize}\n"
    # ))

    cdef double         x, y, r         # Particle x, y, radius
    cdef double         px, py          # Pixel centre x, y

    cdef Py_ssize_t     min_nx, max_nx
    cdef Py_ssize_t     min_ny, max_ny

    cdef Py_ssize_t     num_corners     # Number of intersected corners
    cdef Py_ssize_t[4]  corners         # Intersected corners indices

    cdef Py_ssize_t     i, j, k, l

    for i in range(nrows):
        # Aliases
        x = positions[i, 0]
        y = positions[i, 1]
        r = radii[i]

        if r == 0:
            if x == xlim[1]:
                min_nx = nx - 1
            else:
                min_nx = <Py_ssize_t>floor((x - xlim[0]) / xsize)

            if y == ylim[1]:
                min_ny = ny - 1
            else:
                min_ny = <Py_ssize_t>floor((y - ylim[0]) / ysize)

            if min_nx >= nx or min_nx < 0:
                continue
            if min_ny >= ny or min_ny < 0:
                continue

            pixels[min_nx, min_ny] += 1
            continue

        # Find minimum and maximum pixel indices over which the particle spans.
        # This corresponds to the intersected cells
        min_nx = <Py_ssize_t>floor((x - r - xlim[0]) / xsize)
        max_nx = <Py_ssize_t>floor((x + r - xlim[0]) / xsize)

        min_ny = <Py_ssize_t>floor((y - r - ylim[0]) / ysize)
        max_ny = <Py_ssize_t>floor((y + r - ylim[0]) / ysize)

        # print((
        #     "Before thresholding\n"
        #     f"x = {x}\n"
        #     f"y = {y}\n"
        #     f"r = {r}\n"
        #     f"min_nx = {min_nx}\n"
        #     f"max_nx = {max_nx}\n"
        #     f"min_ny = {min_ny}\n"
        #     f"max_ny = {max_ny}\n"
        # ))

        # Check pixel indices are within the system bounds
        if min_nx >= nx:
            continue
        elif min_nx < 0:
            min_nx = 0

        if max_nx < 0:
            continue
        elif max_nx >= nx:
            max_nx = nx - 1

        if min_ny >= ny:
            continue
        elif min_ny < 0:
            min_ny = 0

        if max_ny < 0:
            continue
        elif max_ny >= ny:
            max_ny = ny - 1

        # print((
        #     "After thresholding\n"
        #     f"x = {x}\n"
        #     f"y = {y}\n"
        #     f"r = {r}\n"
        #     f"min_nx = {min_nx}\n"
        #     f"max_nx = {max_nx}\n"
        #     f"min_ny = {min_ny}\n"
        #     f"max_ny = {max_ny}\n"
        # ))


        # Iterate over all intersected pixels, computing the area covered by
        # the particle
        for j in range(min_nx, max_nx + 1):
            for k in range(min_ny, max_ny + 1):
                # Find the number of pixel corners inside the particle
                num_corners = 0
                for l in range(4):
                    corners[l] = 0

                # Pixel centre coordinates
                px = xlim[0] + (j + 0.5) * xsize
                py = ylim[0] + (k + 0.5) * ysize

                # print((
                #     f"For pixel indices [{j}, {k}]:\n"
                #     f"px = {px}\n"
                #     f"py = {py}\n"
                #     f"r2 = {r * r}\n"
                #     f"dist2 = {dist2(px - 0.5 * xsize, py - 0.5 * ysize, x, y)}\n"
                #     f"dist2 = {dist2(px + 0.5 * xsize, py - 0.5 * ysize, x, y)}\n"
                #     f"dist2 = {dist2(px + 0.5 * xsize, py + 0.5 * ysize, x, y)}\n"
                #     f"dist2 = {dist2(px - 0.5 * xsize, py + 0.5 * ysize, x, y)}\n"
                # ))

                # Lower left corner
                if dist2(px - 0.5 * xsize, py - 0.5 * ysize, x, y) < r * r:
                    num_corners += 1
                    corners[0] = 1

                # Lower right corner
                if dist2(px + 0.5 * xsize, py - 0.5 * ysize, x, y) < r * r:
                    num_corners += 1
                    corners[1] = 1

                # Upper right corner
                if dist2(px + 0.5 * xsize, py + 0.5 * ysize, x, y) < r * r:
                    num_corners += 1
                    corners[2] = 1

                # Upper left corner
                if dist2(px - 0.5 * xsize, py + 0.5 * ysize, x, y) < r * r:
                    num_corners += 1
                    corners[3] = 1

                # print((
                #     f"num_corners = {num_corners}\n"
                #     f"corners = [{corners[0], corners[1], corners[2], corners[3]}]\n"
                # ))

                if num_corners == 0:
                    pixels[j, k] += zero_corner(corners, px, py, xsize, ysize, x, y, r)
                elif num_corners == 1:
                    pixels[j, k] += one_corner(corners, px, py, xsize, ysize, x, y, r)
                elif num_corners == 2:
                    pixels[j, k] += two_corner(corners, px, py, xsize, ysize, x, y, r)
                elif num_corners == 3:
                    pixels[j, k] += three_corner(corners, px, py, xsize, ysize, x, y, r)
                elif num_corners == 4:
                    pixels[j, k] += four_corner(corners, px, py, xsize, ysize, x, y, r)
