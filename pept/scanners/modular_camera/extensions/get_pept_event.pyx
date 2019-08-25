#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : get_pept_event.pyx
# License           : License: GNU v3.0
# Author            : Andrei Leonard Nicusan <aln705@student.bham.ac.uk>
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
