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


# File   : fpi.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 16.04.2021


import  warnings

import  numpy               as      np

from    beartype            import  beartype

import  pept
from    .fpi_ext            import  fpi_ext


class FPI(pept.base.VoxelsFilter):
    '''FPI is a modern voxel-based tracer-location algorithm that can reliably
    work with unknown numbers of tracers in fast and noisy environments.

    It was successfully used to track fast-moving radioactive tracers in pipe
    flows at the Virginia Commonwealth University. If you use this algorithm in
    your work, please cite the following paper:

        Wiggins C, Santos R, Ruggles A. A feature point identification method
        for positron emission particle tracking with multiple tracers. Nuclear
        Instruments and Methods in Physics Research Section A: Accelerators,
        Spectrometers, Detectors and Associated Equipment. 2017 Jan 21;
        843:22-8.

    Permission was granted by Dr. Cody Wiggins in March 2021 to publish his
    code in the `pept` library under the GNU v3.0 license.

    Two main methods are provided: `fit_sample` for tracking a single voxel
    space (i.e. a single `pept.Voxels`) and `fit` which tracks all the samples
    encapsulated in a `pept.VoxelData` class *in parallel*.

    Attributes
    ----------
    w: double
        Search range to be used in local maxima calculation. Typical values for
        w are 2 - 5 (lower number for more particles or smaller particle
        separation).

    r: double
        Fraction of peak value used as threshold. Typical values for r are
        usually between 0.3 and 0.6 (lower for more particles, higher for
        greater background noise)

    lld_counts: double, default 0
        A secondary lld to prevent assigning local maxima to voxels with very
        low values. The parameter lld_counts is not used much in practice -
        for most cases, it can be set to zero.

    Examples
    --------
    A typical workflow would involve reading LoRs from a file, creating a lazy
    `VoxelData` voxellised representation, instantiating an `FPI` class,
    tracking the tracer locations from the LoRs, and plotting them.

    >>> import pept
    >>>
    >>> lors = pept.LineData(...)   # set sample_size and overlap appropriately
    >>> voxels = pept.tracking.Voxelize((50, 50, 50)).fit(lors)
    >>>
    >>> fpi = pept.tracking.FPI(w = 3, r = 0.4)
    >>> positions = fpi.fit(voxels) # this is a `pept.PointData` instance

    A much more efficient approach would be to create a `pept.Pipeline`
    containing a voxelization step and then FPI:

    >>> from pept.tracking import *
    >>>
    >>> pipeline = Voxelize((50, 50, 50)) + FPI() + Stack()
    >>> positions = pipeline.fit(lors)

    Finally, plotting results:

    >>> from pept.plots import PlotlyGrapher
    >>>
    >>> grapher = PlotlyGrapher()
    >>> grapher.add_points(positions)
    >>> grapher.show()

    >>> from pept.plots import PlotlyGrapher2D
    >>> PlotlyGrapher2D().add_timeseries(positions).show()

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.

    '''

    def __init__(
        self,
        w = 3.,
        r = 0.4,
        lld_counts = 0.,
        verbose = False,
    ):
        '''`FPI` class constructor.

        Parameters
        ----------
        w: double
            Search range to be used in local maxima calculation. Typical values
            for w are 2 - 5 (lower number for more particles or smaller
            particle separation).

        r: double
            Fraction of peak value used as threshold. Typical values for r are
            usually between 0.3 and 0.6 (lower for more particles, higher for
            greater background noise)

        lld_counts: double, default 0
            A secondary lld to prevent assigning local maxima to voxels with
            very low values. The parameter `lld_counts` is not used much in
            practice - for most cases, it can be set to zero.

        verbose: bool, default False
            Show extra information on class instantiation.

        '''
        self.w = float(w)
        self.r = float(r)
        self.lld_counts = float(lld_counts)


    @beartype
    def fit_sample(self, voxels: pept.Voxels):
        '''Use the FPI algorithm to locate a tracer from a single voxellised
        space (i.e. from one sample of LoRs).

        A sample of LoRs can be voxellised using the `pept.Voxels.from_lines`
        method before calling this function.

        Parameters
        ----------
        voxels: pept.Voxels
            A single voxellised space (i.e. from a single sample of LoRs) for
            which the tracers' locations will be found using the FPI method.

        timestamp: float, default 0.
            The timestamp to associate with the tracer positions found in this
            voxel space.

        as_array: bool, default False
            If `True`, return the found tracers' locations as a NumPy array.
            Otherwise, return them in a `pept.PointData` instance.

        verbose: bool, default False
            Show extra information on the sample processing step.

        Returns
        -------
        locations: numpy.ndarray or pept.PointData
            The tracked locations found; if `as_array` is True, they are
            returned as a NumPy array with columns [time, x, y, z, error_x,
            error_y, error_z]. If `as_array` is False, the points are returned
            in a `pept.PointData` for ease of visualisation.

        Raises
        ------
        TypeError
            If `voxels` is not an instance of `pept.Voxels` (or subclass
            thereof).
        '''

        points = fpi_ext(
            voxels.voxels,
            self.w,
            self.r,
            self.lld_counts,
        )

        # Insert the time column and translate the coordinates from the voxel
        # space to the physical space
        points[:, 1:4] *= voxels.voxel_size
        points[:, 1:4] += [voxels.xlim[0], voxels.ylim[0], voxels.zlim[0]]

        points = np.insert(points, 0, np.nan, axis = 1)
        if "_lines" in voxels.attrs:
            points[:, 0] = voxels.attrs["_lines"].lines[:, 0].mean()
        else:
            warnings.warn((
                "The input `Voxels` did not have a '_lines' attribute, so no "
                "timestamp can be inferred. The time was set to NaN."
            ), RuntimeWarning)

        return pept.PointData(
            points,
            columns = ["t", "x", "y", "z", "error_x", "error_y", "error_z"],
        )
