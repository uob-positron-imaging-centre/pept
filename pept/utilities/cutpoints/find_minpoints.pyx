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


# File              : find_minpoints.pyx
# License           : GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 20.10.2020


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


import numpy as np      # import numpy for Python functions
cimport numpy as np     # import numpy for C functions (numpy's C API)


np.import_array()


cdef extern from "find_minpoints_ext.c":
    # C is included here so that it doesn't need to be compiled externally
    pass


cdef extern from "find_minpoints_ext.h":
    double* find_minpoints_ext(
        const double *, const Py_ssize_t, const Py_ssize_t, const Py_ssize_t,
        const double, const double *, const int, Py_ssize_t *, Py_ssize_t *
    ) nogil



cpdef find_minpoints(
    const double[:, :] sample_lines,  # LoRs in sample
    const Py_ssize_t num_lines,       # Number of LoRs in groups for computing MDP
    const double max_distance,        # Max allowed distance between two LoRs
    const double[:] cutoffs,          # Spatial cutoff for cutpoints
    bint append_indices = 0           # Append LoR indices used for each cutpoint
):
    '''Compute the minimum distance points (MDPs) from all combinations of
    `num_lines` lines given in an array of lines `sample_lines`.

    ::

        Function signature:
            find_minpoints(
                double[:, :] sample_lines,  # LoRs in sample
                Py_ssize_t num_lines,       # Number of LoRs in combinations
                double max_distance,        # Max distance from MDP to LoRs
                double[:] cutoffs,          # Spatial cutoff for minpoints
                bool append_indices = 0     # Append LoR indices used
            )

    Given a sample of lines, this functions computes the minimum distance
    points (MDPs) for every possible combination of `num_lines` lines. The
    returned numpy array contains all MDPs that satisfy the following:

    1. Are within the `cutoffs`.
    2. Are closer to all the constituent LoRs than `max_distance`.

    Parameters
    ----------
    sample_lines: (M, N) numpy.ndarray
        A 2D array of lines, where each line is defined by two points such that
        every row is formatted as `[t, x1, y1, z1, x2, y2, z2, etc.]`. It
        *must* have at least 2 lines and the combination size `num_lines`
        *must* be smaller or equal to the number of lines. Put differently:
        2 <= num_lines <= len(sample_lines).

    num_lines: int
        The number of lines in each combination of LoRs used to compute the
        MDP. This function considers every combination of `numlines` from the
        input `sample_lines`. It must be smaller or equal to the number of input
        lines `sample_lines`.

    max_distance: float
        The maximum allowed distance between an MDP and its constituent lines.
        If any distance from the MDP to one of its lines is larger than
        `max_distance`, the MDP is thrown away.

    cutoffs: (6,) numpy.ndarray
        An array of spatial cutoff coordinates with *exactly 6 elements* as
        [x_min, x_max, y_min, y_max, z_min, z_max]. If any MDP lies outside
        this region, it is thrown away.

    append_indices: bool
        A boolean specifying whether to include the indices of the lines used
        to compute each MDP. If `False`, the output array will only contain the
        [time, x, y, z] of the MDPs. If `True`, the output array will have
        extra columns [time, x, y, z, line_idx(1), ..., line_idx(n)] where
        n = `num_lines`.

    Returns
    -------
    minpoints: (M, N) numpy.ndarray
        A 2D array of `float`s containing the time and coordinates of the MDPs
        [time, x, y, z]. The time is computed as the average of the constituent
        lines. If `append_indices` is `True`, then `num_lines` indices of the
        constituent lines are appended as extra columns:
        [time, x, y, z, line_idx1, line_idx2, ..].

    Notes
    -----
    There must be at least two lines in `sample_lines` and `num_lines` must be
    greater or equal to the number of lines (i.e. `len(sample_lines)`).
    Put another way: 2 <= num_lines <= len(sample_lines).

    This is a low-level Cython function that does not do any checks on the
    input data - it is meant to be used in other modules / libraries. For a
    normal user, the `pept.tracking.peptml` function `find_minpoints` and
    class `Minpoints` are recommended as higher-level APIs. They do check the
    input data and are easier to use (for example, they automatically compute
    the cutoffs).

    Examples
    --------

    >>> import numpy as np
    >>> from pept.utilities import find_minpoints
    >>>
    >>> lines = np.random.random((500, 7)) * 500
    >>> num_lines = 3
    >>> max_distance = 0.1
    >>> cutoffs = np.array([0, 500, 0, 500, 0, 500], dtype = float)
    >>>
    >>> minpoints = find_minpoints(lines, num_lines, max_distance, cutoffs)

    '''

    # Lines for a single sample => (m, n >= 7) array
    # sample_lines row: [time X1 Y1 Z1 X2 Y2 Z2 etc.]
    cdef Py_ssize_t nrows = sample_lines.shape[0]
    cdef Py_ssize_t ncols = sample_lines.shape[1]

    cdef Py_ssize_t mpts_nrows = 0
    cdef Py_ssize_t mpts_ncols = 0

    cdef double *minpoints
    cdef np.npy_intp[2] size

    with nogil:
        minpoints = find_minpoints_ext(
            &sample_lines[0, 0],
            nrows,
            ncols,
            num_lines,
            max_distance,
            &cutoffs[0],
            append_indices,
            &mpts_nrows,
            &mpts_ncols
        )

    size[0] = mpts_nrows
    size[1] = mpts_ncols

    # Use the `minpoints` pointer as the internal data of a numpy array
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

    cdef np.ndarray[double, ndim=2] mpts_arr = np.PyArray_SimpleNewFromData(
        2, size, np.NPY_FLOAT64, minpoints
    )
    PyArray_ENABLEFLAGS(mpts_arr, np.NPY_OWNDATA)

    # Sort rows based on time (column 0)
    mpts_arr = mpts_arr[mpts_arr[:, 0].argsort()]

    return mpts_arr
