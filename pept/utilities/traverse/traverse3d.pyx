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


# File              : traverse3d.pyx
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
    double[3] u,
    double[3] v,
    double xmin,
    double xmax,
    double ymin,
    double ymax,
    double zmin,
    double zmax,
) nogil:
    '''Given a 3D line defined as L(t) = U + t V and an axis-aligned bounding
    box (AABB), find the `t` for which the line intersects the box, if it
    exists.

    It is assumed that the line starts *outside* the AABB, so that t == 0.0 is
    only reserved for when the line does not intersect the box.
    '''

    cdef double tmin, tmax, tymin, tymax, tzmin, tzmax

    tmin = (xmin - u[0]) / v[0]     # Relies on IEEE FP behaviour for div by 0
    tmax = (xmax - u[0]) / v[0]

    tymin = (ymin - u[1]) / v[1]
    tymax = (ymax - u[1]) / v[1]

    tzmin = (zmin - u[2]) / v[2]
    tzmax = (zmax - u[2]) / v[2]

    if tmin > tmax: swap(&tmin, &tmax)
    if tymin > tymax: swap(&tymin, &tymax)
    if tzmin > tzmax: swap(&tzmin, &tzmax)

    if tmin > tymax or tymin > tmax: return 0.0

    if tymin > tmin: tmin = tymin
    if tymax < tmax: tmax = tymax

    if tmin > tzmax or tzmin > tmax: return 0.0

    if tzmin > tmin: tmin = tzmin
    if tzmax < tmax: tmax = tzmax

    return tmin


cpdef void traverse3d(
    double[:, :, :] voxels,           # Initialised!
    const double[:, :] lines,         # Has exactly 7 columns!
    const double[:] grid_x,           # Has voxels.shape[0] + 1 elements!
    const double[:] grid_y,           # Has voxels.shape[1] + 1 elements!
    const double[:] grid_z            # Has voxels.shape[2] + 1 elements!
) nogil:
    ''' Fast voxel traversal for 3D lines (or LoRs).

    ::

        Function Signature:
            traverse3d(
                long[:, :, :] voxels,       # Initialised!
                double[:, :] lines,         # Has exactly 7 columns!
                double[:] grid_x,           # Has voxels.shape[0] + 1 elements!
                double[:] grid_y,           # Has voxels.shape[1] + 1 elements!
                double[:] grid_z            # Has voxels.shape[2] + 1 elements!
            )

    This function computes the number of lines that passes through each voxel,
    saving the result in `voxels`. It does so in an efficient manner, in which
    for every line, only the voxels that is passes through are traversed.

    As it is highly optimised, this function does not perform any checks on the
    validity of the input data. Please check the parameters before calling
    `traverse3d`, as it WILL segfault on wrong input data. Details are given
    below, along with an example call.

    Parameters
    ----------
    voxels : numpy.ndarray(dtype = numpy.float64, ndim = 3)
        The `voxels` parameter is a numpy.ndarray of shape (X, Y, Z) that
        has been initialised to zeros before the function call. The values
        will be modified in-place in the function to reflect the number of
        lines that pass through each voxel.

    lines : numpy.ndarray(dtype = numpy.float64, ndim = 2)
        The `lines` parameter is a numpy.ndarray of shape(N, 7), where each
        row is formatted as [time, x1, y1, z1, x2, y2, z2]. Only indices 1:7
        will be used as the two points P1 = [x1, y1, z2] and P2 = [x2, y2, z2]
        defining the line (or LoR).

    grid_x : numpy.ndarray(dtype = numpy.float64, ndim = 1)
        The grid_x parameter is a one-dimensional grid that delimits the
        voxels in the x-dimension. It must be *sorted* in ascending order
        with *equally-spaced* numbers and length X + 1 (voxels.shape[0] + 1).

    grid_y : numpy.ndarray(dtype = numpy.float64, ndim = 1)
        The grid_y parameter is a one-dimensional grid that delimits the
        voxels in the y-dimension. It must be *sorted* in ascending order
        with *equally-spaced* numbers and length Y + 1 (voxels.shape[1] + 1).

    grid_z : numpy.ndarray(dtype = numpy.float64, ndim = 1)
        The grid_z parameter is a one-dimensional grid that delimits the
        voxels in the z-dimension. It must be *sorted* in ascending order
        with *equally-spaced* numbers and length Z + 1 (voxels.shape[2] + 1).

    Examples
    --------
    The input parameters can be easily generated using numpy before calling the
    function. For example, if a volume of 300 x 400 x 500 is split into
    30 x 40 x 50 voxels, a possible code would be:

    >>> import numpy as np
    >>> from pept.utilities.traverse import traverse3d
    >>>
    >>> volume = [300, 400, 500]
    >>> number_of_voxels = [30, 40, 50]
    >>> voxels = np.zeros(number_of_voxels)

    The grid has one extra element than the number of voxels. For example, 5
    voxels between 0 and 5 would be delimited by the grid [0, 1, 2, 3, 4, 5]
    which has 6 elements (see off-by-one errors - story of my life).

    >>> grid_x = np.linspace(0, volume[0], number_of_voxels[0] + 1)
    >>> grid_y = np.linspace(0, volume[1], number_of_voxels[1] + 1)
    >>> grid_z = np.linspace(0, volume[2], number_of_voxels[2] + 1)
    >>>
    >>> random_lines = np.random.random((100, 7)) * 300

    Calling `traverse3d` will modify `voxels` in-place.

    >>> traverse3d(voxels, random_lines, grid_x, grid_y, grid_z)

    Notes
    -----
    This function is an adaptation of a widely-used algorithm [1]_, optimised
    for PEPT LoRs traversal.

    .. [1] Amanatides J, Woo A. A fast voxel traversal algorithm for ray tracing.
       InEurographics 1987 Aug 24 (Vol. 87, No. 3, pp. 3-10)..

    '''

    cdef Py_ssize_t n_lines = lines.shape[0]
    cdef Py_ssize_t nx = voxels.shape[0]
    cdef Py_ssize_t ny = voxels.shape[1]
    cdef Py_ssize_t nz = voxels.shape[2]

    # Grid size
    cdef double gsize_x = grid_x[1] - grid_x[0]
    cdef double gsize_y = grid_y[1] - grid_y[0]
    cdef double gsize_z = grid_z[1] - grid_z[0]

    # Delimiting grid
    cdef double xmin = grid_x[0]
    cdef double xmax = grid_x[nx]

    cdef double ymin = grid_y[0]
    cdef double ymax = grid_y[ny]

    cdef double zmin = grid_z[0]
    cdef double zmax = grid_z[nz]

    # The current voxel indices [ix, iy, iz] that the line passes
    # through.
    cdef Py_ssize_t ix, iy, iz

    # Define a line as L(t) = U + t V
    # If an LoR is defined as two points P1 and P2, then
    # U = P1 and V = P2 - P1
    cdef double[3] p1, p2, u, v
    cdef double t

    # The step [step_x, step_y, step_z] defines the sense of the LoR.
    # If V[0] is positive, then step_x = 1
    # If V[0] is negative, then step_x = -1
    cdef Py_ssize_t step_x, step_y, step_z

    # The value of t at which the line passes through to the next
    # voxel, for each dimension.
    cdef double tnext_x, tnext_y, tnext_z

    # deltat indicates how far along the ray we must move (in units of
    # t) for each component to be equal to the size of the voxel in
    # that dimension.
    cdef double deltat_x, deltat_y, deltat_z

    cdef Py_ssize_t i, j

    for i in range(n_lines):
        for j in range(3):
            p1[j] = lines[i, 1 + j]
            p2[j] = lines[i, 4 + j]
            u[j] = p1[j]
            v[j] = p2[j] - p1[j]

        ##############################################################
        # Initialisation stage

        step_x = 1 if v[0] >= 0 else -1
        step_y = 1 if v[1] >= 0 else -1
        step_z = 1 if v[2] >= 0 else -1

        # If the first point is outside the box, find the first voxel it hits
        if (u[0] < xmin or u[0] > xmax or
                u[1] < ymin or u[1] > ymax or
                u[2] < zmin or u[2] > zmax):

            t = intersect(u, v, xmin, xmax, ymin, ymax, zmin, zmax)

            # No intersection
            if t == 0.0:
                continue

            # Overwrite U to correspond to the first intersection point
            for j in range(3):
                u[j] = u[j] + t * v[j]

        # Corner case: every voxel is defined as lower boundary (inclusive) and
        # upper boundary (exclusive). Therefore, at the upper end of the voxel
        # grid an undefined case occurs. If a point lies right at the upper
        # boundary of the voxel space, "move it" a bit lower on the line
        if u[0] == xmax or u[1] == ymax or u[2] == zmax:
            for j in range(3):
                u[j] = u[j] + 1e-5 * v[j]

        # If, for dimension x, there are 5 voxels between coordinates 0
        # and 5, then the delimiting grid is [0, 1, 2, 3, 4, 5].
        # If the line starts at 1.5, then it is part of the voxel at
        # index 1.
        ix = <Py_ssize_t>((u[0] - xmin) / gsize_x)
        iy = <Py_ssize_t>((u[1] - ymin) / gsize_y)
        iz = <Py_ssize_t>((u[2] - zmin) / gsize_z)

        # Check the indices are inside the voxel grid - just to be sure
        if (ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz):
            continue

        # If the line is going "up", the next voxel is the next one
        # If the line is going "down", the next voxel is the current one
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

        if v[2] > 0:
            tnext_z = (grid_z[iz + 1] - u[2]) / v[2]
        elif v[2] < 0:
            tnext_z = (grid_z[iz] - u[2]) / v[2]
        else:
            tnext_z = DBL_MAX

        deltat_x = fabs(gsize_x / v[0]) if v[0] else 0
        deltat_y = fabs(gsize_y / v[1]) if v[1] else 0
        deltat_z = fabs(gsize_z / v[2]) if v[2] else 0

        ###############################################################
        # Incremental traversal stage

        # Loop until we reach the last voxel in space
        while (ix < nx and iy < ny and iz < nz) and (ix >= 0 and iy >= 0 and iz >= 0):

            voxels[ix, iy, iz] += 1.

            # Select the minimum t that makes the line pass
            # through to the next voxel
            if tnext_x < tnext_y:
                if tnext_x < tnext_z:
                    # If the next voxel falls beyond the end of the line (that is at
                    # t = 1), stop the traversal stage
                    if tnext_x > 1.:
                        break

                    ix = ix + step_x
                    tnext_x = tnext_x + deltat_x
                else:
                    if tnext_z > 1.:
                        break

                    iz = iz + step_z
                    tnext_z = tnext_z + deltat_z
            else:
                if tnext_y < tnext_z:
                    if tnext_y > 1.:
                        break

                    iy = iy + step_y
                    tnext_y = tnext_y + deltat_y
                else:
                    if tnext_z > 1.:
                        break

                    iz = iz + step_z
                    tnext_z = tnext_z + deltat_z


