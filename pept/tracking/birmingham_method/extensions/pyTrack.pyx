#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : birmingham_method.pyx
# License           : License: GNU v3.0
# Author            : Sam Manger
# Date              : 21.08.2019

#!python
#cython: language_level=3

cdef extern from "birmingham_method_ext.c":
    # C is included here so that it doesn't need to be compiled externally
    pass

cdef extern from "birmingham_method_ext.h":
    void birmingham_method_ext(const double *, double *, double *, unsigned int, const double)
    
import numpy as np

def birmingham_method(lines,fopt):

    cdef unsigned int n = len(lines)
    lines = np.ravel(lines, order='C')
    
    cdef double fopt_C = fopt

    location = np.zeros(6)
    location = np.ravel(location, order='C')

    used = np.ones(n)
    used = np.ravel(used, order='C')

    cdef double[::1] lines_memview = lines
    cdef double[::1] location_memview = location
    cdef double[::1] used_memview = used

    # print("Beginning C function")
    # print("ninit is: ", n)

    birmingham_method_ext(&lines_memview[0], &location_memview[0], &used_memview[0], n, fopt_C)

    # print("Ended C function")

    return location, used