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


# File              : fpi_ext.pyx
# License           : GNU v3.0
# Author            : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date              : 06.04.2020


# distutils: language=c++

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


cdef extern from "calcPosFPI.hpp":
    double* calcPosFPIC(
        double *voxels,
        Py_ssize_t length,
        Py_ssize_t width,
        Py_ssize_t depth,
        double w,
        double r,
        double lldCounts,
        Py_ssize_t *out_rows,
        Py_ssize_t *out_cols
    ) nogil


cpdef np.ndarray[double, ndim=2] fpi_ext(
    double[:, :, :] voxels,
    double w,
    double r,
    double lldCounts = 0.,
):
    '''Find a tracer's location from a pre-computed voxellised sample of LoRs
    using the Feature Point Identification (FPI) method.

    ::

        Function signature:

            np.ndarray[double, ndim=2] fpi_ext(
                double[:, :, :] voxels,
                double w,
                double r,
                double lldCounts,
            )

    FPI is a modern tracer-location algorithm that was successfully used to
    track fast-moving radioactive tracers in pipe flows at the Virginia
    Commonwealth University. If you use this algorithm in your work, please
    cite the following paper:

        Wiggins C, Santos R, Ruggles A. A feature point identification method
        for positron emission particle tracking with multiple tracers. Nuclear
        Instruments and Methods in Physics Research Section A: Accelerators,
        Spectrometers, Detectors and Associated Equipment. 2017 Jan 21;
        843:22-8.

    Permission was granted explicitly by Dr. Cody Wiggins in March 2021 to
    publish his code in the `pept` library under the GNU v3.0 license.

    The points returned by this function are in *voxel dimensions*, without
    timestamps. They can be translated into physical dimensions and timestamps
    can be added after calling it, e.g. with the `pept.tracking.fpi.FPI`
    class.

    Parameters
    ----------
    voxels: (L, W, D) numpy.ndarray[ndim = 2, dtype = numpy.float64]
        The 3D grid of voxels, initialised to zero. It can be created with
        `numpy.zeros((length, width, depth))`.

    w: double
        Search range to be used in local maxima calculation. Typical values for
        w are 2 - 5 (lower number for more particles or smaller particle
        separation).

    r: double
        Fraction of peak value used as threshold. Typical values for r are
        usually between 0.3 and 0.6 (lower for more particles, higher for
        greater background noise)

    lldCounts: double, default 0
        A secondary lld to prevent assigning local maxima to voxels with very
        low values. The parameter lldCounts is not used much in practice -
        for most cases, it can be set to zero.

    '''

    cdef double *points
    cdef Py_ssize_t nrows = 0
    cdef Py_ssize_t ncols = 0
    cdef np.npy_intp[2] size

    with nogil:
        points = calcPosFPIC(
            &voxels[0, 0, 0],
            voxels.shape[0],
            voxels.shape[1],
            voxels.shape[2],
            w,
            r,
            lldCounts,
            &nrows,
            &ncols,
        )

    size[0] = nrows
    size[1] = ncols

    # Use the `minpoints` pointer as the internal data of a numpy array
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

    cdef np.ndarray[double, ndim=2] points_arr = np.PyArray_SimpleNewFromData(
        2, size, np.NPY_FLOAT64, points
    )
    PyArray_ENABLEFLAGS(points_arr, np.NPY_OWNDATA)

    return points_arr
