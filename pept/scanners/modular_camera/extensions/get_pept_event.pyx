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


# File              : get_pept_event.pyx
# License           : License: GNU v3.0
# Author            : Sam Manger <s.manger@bham.ac.uk>
# Date              : 27.06.2019


#!python
#cython: language_level=3


cdef extern from "get_pept_event_ext.c":
    # C is included here so that it doesn't need to be compiled externally
    pass

cdef extern from "get_pept_event_ext.h":
    void get_pept_event_ext(double *, unsigned int, int, int)
    void get_pept_LOR_ext(double *, unsigned int, int, int)

import numpy as np

def get_pept_event(word, itag, itime):
    # Lines for a single sample => n x 12 array
    # sampleLines row: [word time itag MPnum Bucket1 Bucket2 Block1 Block2 Seg1 Seg2 Plane1 Plane2]

    cdef unsigned int word_C = word
    cdef int itag_C = itag
    cdef int itime_C = itime

    data_array = np.zeros(12, order='C')      # Allocate enough memory
    # data_array = np.ravel(data_array, order='C')              # Cast into 1D array to send to C

    cdef double[::1] data_array_memview = data_array

    get_pept_event_ext(&data_array_memview[0], word_C, itag_C, itime_C)

    return data_array

def get_pept_LOR(word, itag, itime):
    # Lines for a single sample => n x 8 array
    # sampleLines row: [itag time X1 Y1 Z1 X2 Y2 Z2]

    cdef unsigned int word_C = word
    cdef int itag_C = itag
    cdef int itime_C = itime

    LOR = np.zeros(8, order='C')      # Allocate enough memory
    # data_array = np.ravel(data_array, order='C')              # Cast into 1D array to send to C

    cdef double[::1] LOR_memview = LOR

    get_pept_LOR_ext(&LOR_memview[0], word_C, itag_C, itime_C)

    return LOR
