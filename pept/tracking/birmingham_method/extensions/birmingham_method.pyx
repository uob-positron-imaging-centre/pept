#!/usr/bin/env python3
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


# File      : birmingham_method.pyx
# License   : GNU v3.0
# Author    : Sam Manger
# Date      : 21.08.2019


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


import numpy as np      # import numpy for Python functions
cimport numpy as np     # import numpy for C functions (numpy's C API)


cdef extern from "birmingham_method_ext.c":
    # C is included here so that it doesn't need to be compiled externally
    pass


cdef extern from "birmingham_method_ext.h":
    void birmingham_method_ext(
        const double *, const Py_ssize_t, const Py_ssize_t,
        double *, int *, const double
    ) nogil

    void calculate(
        double *, double *, double *, double *, double *, double *,
        double *, double *, double *, double *, double *, double *,
        double *, double *, double *, double *, double *,
        int *, int, int, double *
    ) nogil


# cpdef means it is defined both for Python and C code;
# cdef means it is defined only for C code;

# Cython has a cool function to automatically get memoryviews of the input
# parameters => double[:, :] receives a 2D numpy array.
cpdef birmingham_method(
    const double[:, :] lines,
    const double fopt
):
    '''Use the Birmingham Method to find one tracer location from the LoRs
    stored in `lines`.

    Function signature:
        birmingham_method(
            double[:, :] lines,     # LoRs in a sample
            double fopt             # Fraction of LoRs used to find tracer
        )

    This function receives a numpy array of LoRs (one "sample") from python,
    computing the minimum distance point (MDP). A number of lines that lie
    outside the standard deviation of the MDP are then removed from the set,
    and the MDP is recalculated. This process is repeated until approximately
    a fixed fraction (fopt) of the original lines is left.

    The found tracer position is then returned along with a boolean mask of
    the LoRs that were used to compute it.

    Parameters
    ----------
    lines : (N, M >= 7) numpy.ndarray
        A numpy array of the lines of respones (LoRs) that will be used to find
        a tracer location; each LoR is stored as a timestamp, the 3D
        coordinates of two points defining the line, followed by any additional
        data. The data columns are then `[time, x1, y1, z1, x2, y2, z2, etc]`.
        Note that the extra data is simply ignored by this function.
    fopt : float
        A float number between 0 and 1 representing the fraction of LoRs that
        will be used to compute the tracer location.

    Returns
    -------
    location : (5,) numpy.ndarray
        The computed tracer location, with data columns formatted as
        `[time, x, y, z, error]`.
    used : (N,) numpy.ndarray
        A boolean mask of the LoRs that were used to compute the tracer
        location; that is, a vector of the same length as `lines`, containing 1
        for the rows that were used, and 0 otherwise.

    Notes
    -----
    This is a low-level Cython function that does not do any checks on the
    input data - it is meant to be used in other modules / libraries. For a
    normal user, the `pept.tracking.birmingham_method.BirminghamMethod` class
    methods `fit_sample` and `fit` are recommended as higher-level APIs. They
    do check the input parameters and are easier to use.
    '''

    # Py_ssize_t is the one "strange" type from Cython - it is the type used
    # for indexing arrays, as pointers must have a certain number of bits that
    # is platform-dependent; that type is stored as a C macro in `ssize_t` that
    # Cython also provides as `Py_ssize_t`.
    cdef Py_ssize_t nrows = lines.shape[0]
    cdef Py_ssize_t ncols = lines.shape[1]

    # np.float64 == C double ; np.intc == C int ;
    cdef np.ndarray[double, ndim = 1] location = np.zeros(5, dtype = np.float64)
    cdef np.ndarray[int, ndim = 1] used = np.ones(nrows, dtype = np.intc)

    # Release the GIL as we're in a thread-safe C function for most of our
    # computation time.
    with nogil:
        birmingham_method_ext(
            &lines[0, 0],
            nrows,
            ncols,
            &location[0],
            &used[0],
            fopt
        )

    return location, used


