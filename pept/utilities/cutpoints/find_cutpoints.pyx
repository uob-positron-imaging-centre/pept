# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: infer_types=True
# cython: cdivision=True


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
#    Copyright (C) 2019 Andrei Leonard Nicusan
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


# File              : find_cutpoints.pyx
# License           : License: GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 03.02.2019


import numpy as np


cpdef find_cutpoints(
    double[:, :] sample_lines,
    double max_distance,
    double[:] cutoffs
):
    '''Compute the cutpoints from a given array of lines.

    Function signature:
        find_cutpoints(
            double[:, :] sample_lines,
            double max_distance,
            double[:] cutoffs
        )

    A cutpoint is the point in 3D space that minimises the distance between any
    two lines. For any two non-parallel 3D lines, this point corresponds to
    half the distance between the two lines and is unique.

    This function considers every pair of lines in `sample_lines` and returns
    all the cutpoints that satisfy the following conditions:
        1. The distance between the two lines is smaller than `max_distance`.
        2. The cutpoints is within the `cutoffs`.

    Parameters
    ----------
    sample_lines : (N, 7) numpy.ndarray
        The sample of lines, where each row is [time, x1, y1, z1, x2, y2, z2],
        containing two points [x1, y1, z1] and [x2, y2, z2] defining an LoR.
    max_distance : float
        The maximum distance between two LoRs for their cutpoint to be considered
    cutoffs : (6,) numpy.ndarray
        Only consider the cutpoints that fall within the cutoffs. `cutoffs` has
        the format [min_x, max_x, min_y, max_y, min_z, max_z]

    Returns
    -------
    cutpoints : (M, 4) numpy.ndarray
        A numpy array of the found cutpoints, where each row is [time, x, y, z] for
        the cutpoint. The time is the average between the two LoRs that were used
        to compute the cutpoint. The first column (for time) is sorted.

    Example usage
    -------------
    >>> import numpy as np
    >>> from pept.utilities import find_cutpoints
    >>>
    >>> lines = np.random.random((500, 7)) * 500
    >>> max_distance = 0.1
    >>> cutoffs = np.array([0, 500, 0, 500, 0, 500], dtype = float)
    >>>
    >>> cutpoints = find_cutpoints(lines, max_distance, cutoffs)

    '''
    # Lines for a single sample => n x 7 array
    # sample_lines row: [time X1 Y1 Z1 X2 Y2 Z2]
    cdef int n = sample_lines.shape[0]

    # Pre-allocate enough memory
    # cutpoints row: [time, x, y, z]
    cutpoints = np.zeros((n * (n - 1) // 2, 4), order = 'C')


    cdef double[:, :] cutpoints_mv = cutpoints  # memoryview of cutpoints
    cdef int i, j, k                            # iterators
    cdef int m                                  # cutpoint index
    cdef double[3] P, U, Q, R, QP               # position, direction vectors
    cdef double a, b, c, d, e                   # collected terms
    cdef double denom, s0, t0                   # parameters for lines
    cdef double[3] A0, C0, AC0                  # perpendicular points
    cdef double mx, my, mz                      # cutpoint coordinates

    with nogil:
        m = 0
        for i in range(n - 1):
            for j in range(i + 1, n):

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

                denom = b * b - a * c
                if denom:
                    s0 = (b * e - c * d) / denom
                    t0 = (a * e - b * d) / denom

                    for k in range(3):
                        A0[k] = P[k] + s0 * U[k]
                        C0[k] = Q[k] + t0 * R[k]
                        AC0[k] = A0[k] - C0[k]

                    # Check the distance is smaller than max_distance
                    if (AC0[0] * AC0[0] + AC0[1] * AC0[1] +
                        AC0[2] * AC0[2] < max_distance * max_distance):
                        mx = (A0[0] + C0[0]) / 2
                        my = (A0[1] + C0[1]) / 2
                        mz = (A0[2] + C0[2]) / 2

                        # Check the cutpoint falls within the cutoffs
                        if (mx > cutoffs[0] and mx < cutoffs[1] and
                            my > cutoffs[2] and my < cutoffs[3] and
                            mz > cutoffs[4] and mz < cutoffs[5]):
                            # Average the times of the two lines
                            cutpoints_mv[m, 0] = (sample_lines[i, 0] + sample_lines[j, 0]) / 2
                            cutpoints_mv[m, 1] = mx
                            cutpoints_mv[m, 2] = my
                            cutpoints_mv[m, 3] = mz
                            m = m + 1


    # Truncate the cutpoints which were not written
    cutpoints_truncated = np.delete(cutpoints, slice(m, None, None), 0)
    del(cutpoints)

    # Sort rows based on time (column 0)
    cutpoints_truncated = cutpoints_truncated[cutpoints_truncated[:, 0].argsort()]

    return cutpoints_truncated



