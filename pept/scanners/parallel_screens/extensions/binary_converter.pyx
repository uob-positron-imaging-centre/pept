# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools.
#
#    If you used this codebase or any software making use of it in a scientific
#    publication, we ask you to cite the following paper:
#        Nicuşan AL, Windows-Yule CR. Positron emission particle tracking
#        using machine learning. Review of Scientific Instruments.
#        2020 Jan 1;91(1):013329.
#        https://doi.org/10.1063/1.5129251
#
#    Copyright (C) 2021 the pept developers.
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


# File              : binary_converter.pyx
# License           : GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 01.04.2021


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


cdef extern from "binary_converter_ext.c":
    # C is included here so that it doesn't need to be compiled externally
    pass


cdef extern from "binary_converter_ext.h":
    double* read_adac_binary(const char *, Py_ssize_t *) nogil


cpdef convert_adac_forte(filepath):
    '''Convert an ADAC Forte list mode binary file to a general line data
    format `[time, x1, y1, z1, x2, y2, z2]`, returned as a NumPy array.

    ::

        Function signature:
            binary_converter(filepath)

    Binary converter for the ADAC Forte dual-head gamma camera native list
    mode data. Given the `filepath` to such a binary file (usually with
    extension ".da01"), this function converts the binary contents to the
    general line of response format `[time, x1, y1, z1, x2, y2, z2]`, where
    `z1 = 0` and `z2 = screen_separation` (found from the file).

    The LoRs are returned as a (N, 7) NumPy array, where N is the number of
    LoRs that were found in the file.

    Function parameters
    -------------------
    filepath: str-like
        A string of characters containing the path to the binary file. The
        string's contents are not read in - it will only be used with the
        `fopen` function, so it can contain any characters allowed by the OS
        file system.

    Returns
    -------
    lors: (N, 7) NumPy array
        The 2D array of LoRs, each row containing the time and coordinates of
        the first and second point defining a 3D line, respectively:
        `[time, x1, y1, z1, x2, y2, z2]`.

    Raises
    ------
    FileNotFoundError
        If the `filepath` does not exist or points to an invalid ADAC binary
        file, in which case the C converter subroutine (binary_converter_ext.c)
        prints a specific message.

    Examples
    --------

    >>> import numpy as np
    >>> from pept.scanners.parallel_screens import binary_converter
    >>>
    >>> lines = binary_converter("adac_experiment_data.da01")

    '''

    filepath_utf = str(filepath).encode('UTF-8')

    cdef char                       *filepath_c = filepath_utf

    cdef double                     *lors = NULL
    cdef Py_ssize_t                 lors_elements = 0
    cdef np.npy_intp[2]             shape

    cdef np.ndarray[double, ndim=2] lors_arr

    with nogil:
        lors = read_adac_binary(filepath_c, &lors_elements)

        shape[0] = lors_elements // 7
        shape[1] = 7

    # Use the `lors` pointer as the internal data of a numpy array with
    # PyArray_SimpleNewFromData
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

    if lors is NULL:
        raise FileNotFoundError(
            "Could not convert binary file - see above for error message"
        )
    else:
        lors_arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_FLOAT64, lors)

    PyArray_ENABLEFLAGS(lors_arr, np.NPY_OWNDATA)

    return lors_arr
