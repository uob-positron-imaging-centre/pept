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


# File              : distance_matrix_reachable.pyx
# License           : GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 10.06.2020


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


import numpy as np
from scipy.sparse import csr_matrix

from libc.float cimport DBL_MIN
from libc.math cimport sqrt
cimport numpy as np


cpdef distance_matrix_reachable(
    const double[:, :] pts,           # Array of points, cols = [t, x, y, z, ...]
    const int points_window
):
    '''Compute the distance matrix from a time-sorted array of points `pts`
    based on a sliding `points_window`.

    ::

        Function signature:
            distance_matrix_reachable(
                double[:, :] pts,   # Array of points, cols = [t, x, y, z, ...]
                int points_window
            )

    The distance between the points (pts[i], pts[j]) is stored in the distance
    matrix at indices (i, j), making it upper-triangular.

    The distance matrix is created and returned using SciPy's sparse CSR
    matrix format. This saves a lot in terms of memory usage, especially for
    time-series data such as moving tracers, as not all points can be
    connected. This format is also closer to the mathematical formulation of
    "reachable points" in terms of undirected (incomplete) graphs - namely
    storing edges as a list of pairs of vertices.

    This is a low-level Cython function that does not do any checks on the
    input data - it is meant to be used in other modules / libraries; in
    particular, the `pept.tracking.trajectory_separation` module.

    Parameters
    ----------
    pts : (M, N>=4) numpy.ndarray
        The points from multiple trajectories. Each row in `pts` will
        have a timestamp and the 3 spatial coordinates, such that the data
        columns are [time, x_coord, y_coord, z_coord]. Note that `point_data`
        can have more data columns and they will simply be ignored.
    points_window : int
        Two points are "reachable" (i.e. they can be connected) if and only if
        they are within `points_window` in the time-sorted input `pts`. As the
        points from different trajectories are intertwined (e.g. for two
        tracers A and B, the `pts` array might have two entries for A,
        followed by three entries for B, then one entry for A, etc.), this
        should optimally be the largest number of points in the input array
        between two consecutive points on the same trajectory. If
        `pts` is too small, all points in the dataset will be unreachable.
        Naturally, a larger `time_window` correponds to more pairs needing to
        be checked (and the function will take a longer to complete).

    Returns
    -------
    distance_matrix : CSR <NxN sparse matrix of type '<class 'numpy.float64'>
        A SciPy sparse matrix in the CSR format, containing the distances
        between every pair of reachable points in `pts`.

    Notes
    -----
    In order for the `points_window` to act as a sliding window, in effect only
    connecting points which are around the same timeframe, the points should be
    sorted based on thetime column (the first row) in `pts`. This should be
    done *prior* to calling this function.
    '''

    # Use Py_ssize_t as we will access C arrays (memoryviews on numpy arrays).
    # That is the "proper" type of a C array pointer / index.
    cdef Py_ssize_t n = pts.shape[0]            # Total number of points
    cdef Py_ssize_t p = min(n, points_window)

    # Calculate sparse distance matrix between reachable points. The number of
    # points we need to check `ndists` is given by the formula below.
    cdef Py_ssize_t ndists = (p + 1) * (n - p) + p * (p + 1) // 2 - n

    # Pre-allocate the arrays for creating the sparse distance matrix. In the
    # sparse matrix, every data point `dists` has an associated row in `rows`
    # column in `cols`.
    cdef np.ndarray[double, ndim = 1] dists_arr = np.zeros(ndists, dtype =
                                                           np.float64)
    cdef np.ndarray[double, ndim = 1] rows_arr = np.zeros(ndists, dtype =
                                                          np.float64)
    cdef np.ndarray[double, ndim = 1] cols_arr = np.zeros(ndists, dtype =
                                                          np.float64)

    # We'll work with memoryviews on the above arrays:
    cdef double[:] dists = dists_arr
    cdef double[:] rows = rows_arr
    cdef double[:] cols = cols_arr

    # Calculate the distances between reachable points.
    cdef Py_ssize_t ie = 0  # distance index
    cdef Py_ssize_t i, j    # iterators
    cdef double dist        # distance between two points

    with nogil:
        for i in range(n - 1):
            for j in range(i + 1, min(i + p, n - 1) + 1):
                # Euclidean distance between points i, j in `pts`
                # dist = np.linalg.norm(pts[i, 1:4] - pts[j, 1:4])
                dist = sqrt(
                    (pts[j, 1] - pts[i, 1]) ** 2 +
                    (pts[j, 2] - pts[i, 2]) ** 2 +
                    (pts[j, 3] - pts[i, 3]) ** 2
                )

                # Fix bug (or feature?) of scipy's minimum_spanning_tree where
                # duplicate points (i.e. dist == 0.0) are ommitted from the
                # MST vertices.
                dists[ie] = dist if dist != 0.0 else DBL_MIN
                rows[ie] = i
                cols[ie] = j
                ie = ie + 1

    # Create the distance matrix from the found points.
    distance_matrix = csr_matrix(
        (dists, (rows, cols)),
        shape = (n, n)
    )

    return distance_matrix
