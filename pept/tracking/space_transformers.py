#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : space_transformers.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 11.08.2021


import textwrap

import  numpy           as      np
import  numba           as      nb

from    pept            import  PointData, LineData, Voxels
from    pept.base       import  LineDataFilter, PointDataFilter

from    .transformers   import  Stack




class Voxelliser(LineDataFilter):

    def __init__(
        self,
        number_of_voxels,
        xlim = None,
        ylim = None,
        zlim = None,
        set_lims = None,
    ):
        # Type-checking inputs
        number_of_voxels = np.asarray(
            number_of_voxels,
            order = "C",
            dtype = int
        )

        if number_of_voxels.ndim != 1 or len(number_of_voxels) != 3:
            raise ValueError(textwrap.fill((
                "The input `number_of_voxels` must be a list-like "
                "with exactly three values, corresponding to the "
                "number of voxels in the x-, y-, and z-dimension. "
                f"Received parameter with shape {number_of_voxels.shape}."
            )))

        if (number_of_voxels < 2).any():
            raise ValueError(textwrap.fill((
                "The input `number_of_voxels` must set at least two "
                "voxels in each dimension (i.e. all elements in "
                "`number_of_elements` must be larger or equal to two). "
                f"Received `{number_of_voxels}`."
            )))

        # Keep track of limits we have to set
        set_xlim = True
        set_ylim = True
        set_zlim = True

        if xlim is not None:
            set_xlim = False
            xlim = np.asarray(xlim, dtype = float)

            if xlim.ndim != 1 or len(xlim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `xlim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the voxel space in the x-dimension. "
                    f"Received parameter with shape {xlim.shape}."
                )))

        if ylim is not None:
            set_ylim = False
            ylim = np.asarray(ylim, dtype = float)

            if ylim.ndim != 1 or len(ylim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `ylim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the voxel space in the y-dimension. "
                    f"Received parameter with shape {ylim.shape}."
                )))

        if zlim is not None:
            set_zlim = False
            zlim = np.asarray(zlim, dtype = float)

            if zlim.ndim != 1 or len(zlim) != 2:
                raise ValueError(textwrap.fill((
                    "The input `zlim` parameter must be a list with exactly "
                    "two values, corresponding to the minimum and maximum "
                    "coordinates of the voxel space in the z-dimension. "
                    f"Received parameter with shape {ylim.shape}."
                )))

        # Setting class attributes
        self._number_of_voxels = tuple(number_of_voxels)
        self._xlim = xlim
        self._ylim = ylim
        self._zlim = zlim

        if set_lims is not None:
            if isinstance(set_lims, LineData):
                self.set_lims(set_lims.lines, set_xlim, set_ylim, set_zlim)
            else:
                self.set_lims(set_lims, set_xlim, set_ylim, set_zlim)


    def set_lims(
        self,
        lines,
        set_xlim = True,
        set_ylim = True,
        set_zlim = True,
    ):
        lines = np.asarray(lines, dtype = float, order = "C")

        if set_xlim:
            xlim = Voxels.get_cutoff(lines[:, 1], lines[:, 4])

        if set_ylim:
            ylim = Voxels.get_cutoff(lines[:, 2], lines[:, 5])

        if set_zlim:
            zlim = Voxels.get_cutoff(lines[:, 3], lines[:, 6])

        self._xlim = xlim
        self._ylim = ylim
        self._zlim = zlim


    @property
    def number_of_voxels(self):
        return self._number_of_voxels


    @property
    def xlim(self):
        return self._xlim


    @property
    def ylim(self):
        return self._ylim


    @property
    def zlim(self):
        return self._zlim


    def fit_sample(self, sample_lines):
        if isinstance(sample_lines, LineData):
            sample_lines = sample_lines.lines

        return Voxels.from_lines(
            sample_lines,
            self.number_of_voxels,
            self.xlim,
            self.ylim,
            self.zlim,
            verbose = False,
        )




@nb.jit
def polyfit(x, y, deg):
    '''Fit polynomial of order `deg` against x, y data points.'''
    mat = np.zeros(shape = (x.shape[0], deg + 1))
    mat[:, 0] = np.ones_like(x)
    for n in range(1, deg + 1):
        mat[:, n] = x**n

    p = np.linalg.lstsq(mat, y)[0]
    return p


@nb.jit
def polyder(p):
    '''Differentiate polynomial p.'''
    d = np.zeros(shape = (p.shape[0] - 1))
    for n in range(d.shape[0]):
        d[n] = (n + 1) * p[n + 1]
    return d


@nb.jit
def polyval(p, x):
    '''Evaluate polynomial p(x) using Horner's Method.

    New numpy.polynomial.Polynomial format:
        p[0] + p[1] * x + p[2] * x^2 + ...
    '''
    result = np.zeros_like(x)
    for coeff in p[::-1]:
        result = x * result + coeff
    return result


@nb.jit
def compute_velocities(points, window, deg):
    # Pre-allocate velocities matrix, columns [vx, vy, vz]
    v = np.zeros((points.shape[0], 3))

    # Half-window size
    hw = (window - 1) // 2

    # Infer velocities of first hw points
    w = points[:window]
    vx = polyder(polyfit(w[:, 0], w[:, 1], deg))
    vy = polyder(polyfit(w[:, 0], w[:, 2], deg))
    vz = polyder(polyfit(w[:, 0], w[:, 3], deg))

    for i in range(hw + 1):
        v[i, 0] = polyval(vx, w[i, 0:1])[0]
        v[i, 1] = polyval(vy, w[i, 0:1])[0]
        v[i, 2] = polyval(vz, w[i, 0:1])[0]

    # Compute velocities in a sliding window
    for i in range(hw + 1, points.shape[0] - hw):
        w = points[i - hw:i + hw + 1]

        vx = polyder(polyfit(w[:, 0], w[:, 1], deg))
        vy = polyder(polyfit(w[:, 0], w[:, 2], deg))
        vz = polyder(polyfit(w[:, 0], w[:, 3], deg))

        v[i, 0] = polyval(vx, points[i, 0:1])[0]
        v[i, 1] = polyval(vy, points[i, 0:1])[0]
        v[i, 2] = polyval(vz, points[i, 0:1])[0]

    # Infer velocities of last hw points
    for i in range(points.shape[0] - hw, points.shape[0]):
        v[i, 0] = polyval(vx, points[i, 0:1])[0]
        v[i, 1] = polyval(vy, points[i, 0:1])[0]
        v[i, 2] = polyval(vz, points[i, 0:1])[0]

    return v




class Velocity(PointDataFilter):

    def __init__(self, window, degree = 2, absolute = False):
        self.window = int(window)
        assert self.window % 2 == 1, "The `window` must be an odd number!"
        self.degree = int(degree)
        self.absolute = bool(absolute)


    def fit_sample(self, samples):
        if not isinstance(samples, PointData):
            samples = PointData(samples)

        vels = compute_velocities(samples.points, self.window, self.degree)

        # Create new object like sample with the extra velocity columns
        if self.absolute:
            absvels = np.linalg.norm(vels, axis = 1)
            points = np.c_[samples.points, absvels]
            columns = samples.columns + ["v"]
        else:
            points = np.c_[samples.points, vels]
            columns = samples.columns + ["vx", "vy", "vz"]

        new_sample = samples.copy(data = points, columns = columns)

        return new_sample
