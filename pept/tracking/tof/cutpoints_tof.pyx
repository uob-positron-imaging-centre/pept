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


# File              : find_cutpoints_tof.pyx
# License           : GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 30.09.2021


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


import numpy as np


cpdef find_cutpoints_tof(
    const double[:, :] sample_lines,  # LoRs in sample
    const double[:, :] tofpoints,     # ToF-predicted points for each LoR
    const double max_distance,        # Max allowed distance between two tofpoints
    const double[:] cutoffs,          # Spatial cutoff for cutpoints
    bint append_indices = 0           # Append LoR indices used for each cutpoint
):
    '''Compute the cutpoints from a given array of lines.

    ::

        Function signature:
            find_cutpoints(
                double[:, :] sample_lines,  # LoRs in sample
                double[:, :] tofpoints,     # ToF-predicted points
                double max_distance,        # Max distance between two LoRs
                double[:] cutoffs,          # Spatial cutoff for cutpoints
                bint append_indices = False # Append LoR indices used
            )

    This is a low-level Cython function that does not do any checks on the
    input data - it is meant to be used in other modules / libraries. For a
    normal user, the `pept.tracking` class `CutpointsToF` is recommended as
    higher-level APIs. They do check the input data and are easier to use (for
    example, they automatically compute the cutoffs).

    A cutpoint is the point in 3D space that minimises the distance between any
    two lines. For any two non-parallel 3D lines, this point corresponds to the
    midpoint of the unique segment that is perpendicular to both lines.

    This function considers every pair of lines in `sample_lines` and returns
    all the cutpoints that satisfy the following conditions:

    1. The distance between the ToF-predicted points on each line is smaller
       than `max_distance`.
    2. The cutpoints are within the `cutoffs`.

    Parameters
    ----------
    sample_lines : (N, M >= 7) numpy.ndarray
        The sample of lines, where each row is [time, x1, y1, z1, x2, y2, z2],
        containing two points [x1, y1, z1] and [x2, y2, z2] defining an LoR.
    tofpoints : (N, M >= 4) numpy.ndarray
        The time of flight-predicted annihilation points on each corresponding
        line, formatted as [time, x, y, z] - has the same number of rows as
        `sample_lines`.
    max_distance : float
        The maximum distance between two LoRs for their cutpoint to be
        considered.
    cutoffs : (6,) numpy.ndarray
        Only consider the cutpoints that fall within the cutoffs. `cutoffs` has
        the format [min_x, max_x, min_y, max_y, min_z, max_z].
    append_indices : bool, optional
        If set to `True`, the indices of the individual LoRs that were used
        to compute each cutpoint is also appended to the returned array.
        Default is `False`.

    Returns
    -------
    cutpoints : (M, 4) or (M, 6) numpy.ndarray
        A numpy array of the calculated weighted cutpoints. If `append_indices`
        is `False`, then the columns are [time, x, y, z]. If `append_indices`
        is `True`, then the columns are [time, x, y, z, i, j], where `i` and
        `j` are the LoR indices from `sample_lines` that were used to compute
        the cutpoints. The time is the average between the timestamps of the
        two LoRs that were used to compute the cutpoint. The first column (for
        time) is sorted.

    '''

    # Lines for a single sample => n x 7 array
    # sample_lines row: [time X1 Y1 Z1 X2 Y2 Z2]
    cdef Py_ssize_t n = sample_lines.shape[0]

    # Pre-allocate enough memory
    # weighted_cutpoints cols: [time, x, y, z, [LoR1_index, LoR2_index]]
    if append_indices:
        cutpoints = np.zeros((n * (n - 1) // 2, 6), order = 'C')
    else:
        cutpoints = np.zeros((n * (n - 1) // 2, 4), order = 'C')

    cdef double[:, :] cutpoints_mv = cutpoints  # memoryview of cutpoints
    cdef Py_ssize_t i, j, k                     # iterators
    cdef Py_ssize_t ic                          # cutpoint index
    cdef double[3] tof                          # tofpoints' distance
    cdef double[3] P, U, Q, R, QP               # position, direction vectors
    cdef double a, b, c, d, e                   # collected terms
    cdef double denom, s0, t0                   # parameters for lines
    cdef double[3] A0, C0                       # perpendicular points
    cdef double mx, my, mz                      # cutpoint coordinates

    with nogil:
        ic = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Check the distance between the two lines' tofpoints is
                # smaller than max_distance
                for k in range(3):
                    tof[k] = tofpoints[i, 1 + k] - tofpoints[j, 1 + k]
                if not (tof[0] * tof[0] +
                        tof[1] * tof[1] +
                        tof[2] * tof[2] < max_distance * max_distance):
                    continue

                # Write each pair of lines in terms of a position vector and a
                # direction vector:
                # L1 : A(s) = P + s U
                # L2 : C(t) = Q + t R
                for k in range(3):
                    P[k] = sample_lines[i, 1 + k]
                    U[k] = sample_lines[i, 4 + k] - P[k]
                    Q[k] = sample_lines[j, 1 + k]
                    R[k] = sample_lines[j, 4 + k] - Q[k]
                    QP[k] = Q[k] - P[k]

                a = U[0] * U[0] + U[1] * U[1] + U[2] * U[2]
                b = U[0] * R[0] + U[1] * R[1] + U[2] * R[2]
                c = R[0] * R[0] + R[1] * R[1] + R[2] * R[2]
                d = U[0] * QP[0] + U[1] * QP[1] + U[2] * QP[2]
                e = QP[0] * R[0] + QP[1] * R[1] + QP[2] * R[2]

                # Check lines are not perfectly parallel
                denom = b * b - a * c
                if not denom:
                    continue

                s0 = (b * e - c * d) / denom
                t0 = (a * e - b * d) / denom

                for k in range(3):
                    A0[k] = P[k] + s0 * U[k]
                    C0[k] = Q[k] + t0 * R[k]

                mx = (A0[0] + C0[0]) / 2
                my = (A0[1] + C0[1]) / 2
                mz = (A0[2] + C0[2]) / 2

                # Check the cutpoint falls within the cutoffs
                if not (mx > cutoffs[0] and mx < cutoffs[1] and
                        my > cutoffs[2] and my < cutoffs[3] and
                        mz > cutoffs[4] and mz < cutoffs[5]):
                    continue

                # Average the times of the two lines
                cutpoints_mv[ic, 0] = (sample_lines[i, 0] + \
                                       sample_lines[j, 0]) / 2
                cutpoints_mv[ic, 1] = mx
                cutpoints_mv[ic, 2] = my
                cutpoints_mv[ic, 3] = mz

                if append_indices:
                    cutpoints_mv[ic, 4] = <double>i
                    cutpoints_mv[ic, 5] = <double>j

                ic = ic + 1

    # Truncate the cutpoints which were not written
    cutpoints_truncated = np.delete(cutpoints, slice(ic, None, None), 0)
    del(cutpoints)

    # Sort rows based on time (column 0)
    cutpoints_truncated = cutpoints_truncated[cutpoints_truncated[:, 0].argsort()]

    return cutpoints_truncated
