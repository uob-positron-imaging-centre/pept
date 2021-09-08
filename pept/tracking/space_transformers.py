#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : space_transformers.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 11.08.2021


import  textwrap

import  numpy               as      np
from    scipy.interpolate   import  interp1d

from    pept                import  LineData, PointData, Voxels
from    pept.base           import  LineDataFilter, PointDataFilter




class Voxelize(LineDataFilter):
    '''Asynchronously voxelize samples of lines from a `pept.LineData`.

    Filter signature:

    ::

        LineData -> Voxelize.fit_sample -> PointData

    This filter is much more memory-efficient than voxelizing all samples of
    LoRs at once - which often overflows the available memory. Most often this
    is used alongside voxel-based tracking algorithms, e.g.
    ``pept.tracking.FPI``:

    >>> from pept.tracking import *
    >>> pipeline = pept.Pipeline([
    >>>     Voxelize((50, 50, 50)),
    >>>     FPI(3, 0.4),
    >>>     Stack(),
    >>> ])

    Parameters
    ----------
    number_of_voxels : 3-tuple
        A tuple-like containing exactly three integers specifying the number of
        voxels to be used in each dimension.

    xlim : (2,) list[float], optional
        The lower and upper boundaries of the voxellised volume in the
        x-dimension, formatted as [x_min, x_max]. If undefined, it is
        inferred from the bounding box of each sample of lines.

    ylim : (2,) list[float], optional
        The lower and upper boundaries of the voxellised volume in the
        y-dimension, formatted as [y_min, y_max]. If undefined, it is
        inferred from the bounding box of each sample of lines.

    zlim : (2,) list[float], optional
        The lower and upper boundaries of the voxellised volume in the
        z-dimension, formatted as [z_min, z_max]. If undefined, it is
        inferred from the bounding box of each sample of lines.

    set_lines : (N, 7) numpy.ndarray or pept.LineData, optional
        If defined, set the system limits upon creating the class to the
        bounding box of the lines in `set_lines`.

    '''

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
        if not isinstance(sample_lines, LineData):
            sample_lines = LineData(sample_lines)

        vox = Voxels.from_lines(
            sample_lines.lines,
            self.number_of_voxels,
            self.xlim,
            self.ylim,
            self.zlim,
            verbose = False,
        )

        # Save the constituent LoRs as a hidden attribute
        vox.attrs["_lines"] = sample_lines
        return vox




class Interpolate(PointDataFilter):
    '''Interpolate between data points at a fixed sampling rate; useful for
    Eulerian fields computation.

    Filter signature:

    ::

        PointData -> Interpolate.fit_sample -> PointData

    By default, the linear interpolator `scipy.interpolate.interp1d` is used.
    You can specify a different interpolator and keyword arguments to send it.
    E.g. nearest-neighbour interpolation: ``Interpolate(1., kind = "nearest")``
    or cubic interpolation: ``Interpolate(1., kind = "cubic")``.

    All data columns except timestamps are interpolated.
    '''

    def __init__(self, timestep, interpolator = interp1d, **kwargs):
        self.timestep = float(timestep)
        self.interpolator = interpolator
        self.kwargs = kwargs


    def fit_sample(self, sample):
        if not isinstance(sample, PointData):
            sample = PointData(sample)

        if not len(sample):
            return sample[0:0]

        # Create interpolators for each dimension (column) wrt time
        interps = [
            self.interpolator(
                sample.points[:, 0],
                sample.points[:, i],
                **self.kwargs,
            ) for i in range(1, sample.points.shape[1])
        ]

        # Sampling points, between the first and last timestamps
        sampling = np.arange(sample.points[0, 0], sample.points[-1, 0],
                             self.timestep)

        data = np.vstack([sampling] + [interp(sampling) for interp in interps])
        return sample.copy(data = data.T)
