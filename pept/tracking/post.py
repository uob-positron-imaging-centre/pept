#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : post.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 06.09.2021


import  warnings

import  numpy           as      np

try:
    class NotInstalled:
        '''Class used to signal that Numba is not available.'''
        pass

    import  numba       as      nb
except ImportError:
    nb = NotInstalled()
    nb.njit = lambda func: func

from    pept            import  PointData
from    pept.base       import  PointDataFilter


@nb.njit
def polyfit(x, y, deg):
    '''Fit polynomial of order `deg` against x, y data points.'''
    mat = np.zeros(shape = (x.shape[0], deg + 1))
    mat[:, 0] = np.ones_like(x)
    for n in range(1, deg + 1):
        mat[:, n] = x**n

    p = np.linalg.lstsq(mat, y)[0]
    return p


@nb.njit
def polyder(p):
    '''Differentiate polynomial p.'''
    d = np.zeros(shape = (p.shape[0] - 1))
    for n in range(d.shape[0]):
        d[n] = (n + 1) * p[n + 1]
    return d


@nb.njit
def polyval(p, x):
    '''Evaluate polynomial p(x) using Horner's Method.

    New numpy.polynomial.Polynomial format:
        p[0] + p[1] * x + p[2] * x^2 + ...
    '''
    result = np.zeros_like(x)
    for coeff in p[::-1]:
        result = x * result + coeff
    return result


@nb.njit
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
    '''Append the dimension-wise or absolute velocity to samples of points
    using a 2D fitted polynomial in a rolling window mode.

    Filter signature:

    ::

        PointData -> Velocity.fit_sample -> PointData

    If Numba is installed, a fast, natively-compiled algorithm is used.

    If `absolute = False`, the "vx", "vy" and "vz" columns are appended. If
    `absolute = True`, then the "v" column is appended.
    '''

    def __init__(self, window, degree = 2, absolute = False):
        if isinstance(nb, NotInstalled):
            warnings.warn((
                "Numba is not installed, so this function will be very slow. "
                "Install Numba to JIT compile the compute-intensive part."
            ), RuntimeWarning)

        self.window = int(window)
        assert self.window % 2 == 1, "The `window` must be an odd number!"
        self.degree = int(degree)
        self.absolute = bool(absolute)

        assert self.window > self.degree, "The `window` must be >`degree`!"


    def fit_sample(self, samples):
        if not isinstance(samples, PointData):
            samples = PointData(samples)

        if not len(samples.points):
            return self._empty_sample(samples)

        if self.window >= len(samples.points):
            return self._invalid_sample(samples)

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


    def _empty_sample(self, samples):
        if self.absolute:
            columns = samples.columns + ["v"]
        else:
            columns = samples.columns + ["vx", "vy", "vz"]

        return samples.copy(
            data = np.empty((0, len(columns))),
            columns = columns,
        )


    def _invalid_sample(self, samples):
        if self.absolute:
            columns = samples.columns + ["v"]
            vels = np.full(len(samples.points), np.nan)
        else:
            columns = samples.columns + ["vx", "vy", "vz"]
            vels = np.full((len(samples.points), 3), np.nan)

        return samples.copy(
            data = np.c_[samples.points, vels],
            columns = columns,
        )
