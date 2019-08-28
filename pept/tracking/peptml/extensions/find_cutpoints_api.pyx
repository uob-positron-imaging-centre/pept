#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
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


# File              : find_cutpoints_api.pyx
# License           : License: GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 27.06.2019


#!python
#cython: language_level=3


cdef extern from "find_cutpoints_ext.c":
    # C is included here so that it doesn't need to be compiled externally
    pass

cdef extern from "find_cutpoints_ext.h":
    void find_cutpoints_ext(const double *, double *, const unsigned int, const double)


import numpy as np

def find_cutpoints_api(sample_lines, max_distance):
    # Lines for a single sample => n x 7 array
    # sample_lines row: [time X1 Y1 Z1 X2 Y2 Z2]
    if sample_lines.ndim != 2 or sample_lines.shape[1] != 7:
        raise ValueError("Expected sample_lines to have shape (n, 7). Received {}".format(sample_lines.shape))

    cdef double max_distance_c = max_distance
    cdef unsigned int n = len(sample_lines)
    # Cast into 1D array to send to C
    sample_lines = np.ravel(sample_lines).astype(float, order = 'C', copy = False)

    # cutpoints row: [TimeM, XM, YM, ZM]

    # Allocate enough memory
    cutpoints = np.zeros((n * (n - 1) // 2, 4), order = 'C')
    # Cast into 1D array to send to C
    cutpoints = np.ravel(cutpoints, order = 'C')

    cdef double[::1] sample_lines_memview = sample_lines
    cdef double[::1] cutpoints_memview = cutpoints

    find_cutpoints_ext(&sample_lines_memview[0], &cutpoints_memview[0], n, max_distance_c)
    cutpoints = cutpoints.reshape((-1, 4))

    # cut all rows which have zero values
    cutpoints = cutpoints[np.all(cutpoints, axis = 1)]

    # sort rows based on time (column 0)
    cutpoints = cutpoints[cutpoints[:,0].argsort()]

    return cutpoints

