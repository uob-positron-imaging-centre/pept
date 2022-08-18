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
#    Copyright (C) 2019-2022 the pept developers
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


# File              : sampling_extensions.pyx
# License           : GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 14.08.2022


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


import numpy as np      # import numpy for Python functions

from libc.stdint cimport int64_t


cpdef samples_indices_adaptive_window_ext(
    const double[:, :] data,          # LoRs in sample
    const double window,              # Time window
    const double overlap,             # Time window overlap
    const Py_ssize_t max_elems,       # Maximum number of LoRs in a sample
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
    sample_lines : (M, N) numpy.ndarray
        A 2D array of lines, where each line is defined by two points such that
        every row is formatted as `[t, x1, y1, z1, x2, y2, z2, etc.]`. It
        *must* have at least 2 lines and the combination size `num_lines`
        *must* be smaller or equal to the number of lines. Put differently:
        2 <= num_lines <= len(sample_lines).

    num_lines : int
        The number of lines in each combination of LoRs used to compute the
        MDP. This function considers every combination of `numlines` from the
        input `sample_lines`. It must be smaller or equal to the number of input
        lines `sample_lines`.

    max_distance : float
        The maximum allowed distance between an MDP and its constituent lines.
        If any distance from the MDP to one of its lines is larger than
        `max_distance`, the MDP is thrown away.

    cutoffs : (6,) numpy.ndarray
        An array of spatial cutoff coordinates with *exactly 6 elements* as
        [x_min, x_max, y_min, y_max, z_min, z_max]. If any MDP lies outside
        this region, it is thrown away.

    append_indices : bool
        A boolean specifying whether to include the indices of the lines used
        to compute each MDP. If `False`, the output array will only contain the
        [time, x, y, z] of the MDPs. If `True`, the output array will have
        extra columns [time, x, y, z, line_idx(1), ..., line_idx(n)] where
        n = `num_lines`.

    Returns
    -------
    minpoints : (M, N) numpy.ndarray
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

    if overlap >= window or max_elems < 1:
        raise ValueError("overlap >= window || max_elems < 1")

    # Approximate number of samples - will still need to check / reallocate as needed
    samples_indices_arr = np.zeros(
        (<int>((data[len(data) - 1, 0] - data[0, 0]) / (window - overlap)), 2),
        dtype = np.int64,
    )

    cdef int64_t[:, :] samples_indices = samples_indices_arr

    # Index and time of the start of the current window
    cdef int64_t istart = 0
    cdef double  tstart = data[0, 0]

    # Index and time of the start of the next window, considering overlap
    cdef int64_t inext = 0
    cdef double  tnext = tstart + window - overlap
    cdef int64_t found_next = 0

    # Line index, sample index, number of lines
    cdef int64_t i = 0
    cdef int64_t isample = 0
    cdef int64_t nrows = data.shape[0]

    while i < nrows:
        # print(f"t={data[i, 0]}")
        # print(f"{i=}\n{isample=}\n{istart=}\n{tstart=}\n{inext=}\n{tnext=}\n")

        # If next window is found (considering overlap)
        if data[i, 0] > tnext and found_next == 0:
            inext = i
            found_next = 1

        # If the end of the current window is found - i.e. time window is
        # exceeded or maximum number of lines is reached
        if data[i, 0] > tstart + window or i - istart >= max_elems:

            # If the next window's start wasn't reached yet, just start here
            if found_next == 0:
                inext = i

            # Set the end index of the current sample
            samples_indices[isample, 1] = i
            isample += 1

            # Move indices to next window's start
            istart = inext
            i = istart

            # The starting time of the next sample
            tstart = data[istart, 0]
            tnext = tstart + window - overlap

            # Reallocate samples_indices if more samples are needed
            if isample >= len(samples_indices):
                samples_indices_new = np.zeros(
                    (2 * len(samples_indices), 2),
                    dtype = np.int64,
                )

                samples_indices_new[:len(samples_indices), :] = samples_indices
                samples_indices_arr = samples_indices_new
                samples_indices = samples_indices_arr

            samples_indices[isample, 0] = istart
            found_next = 0

        i += 1

    samples_indices[isample, 1] = i

    # Truncate the samples indices which were not written to
    truncated = np.delete(samples_indices_arr, slice(isample + 1, None, None), 0)
    del(samples_indices_arr)

    return truncated
