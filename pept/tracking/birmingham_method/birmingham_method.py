#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
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


# File   : birmingham_method.py
# License: GNU v3.0
# Author : Sam Manger <s.manger@bham.ac.uk>
# Date   : 20.08.2019



from    beartype                        import  beartype
import  numpy                           as      np

import  pept

from    .extensions.birmingham_method   import  birmingham_method


class BirminghamMethod(pept.base.LineDataFilter):
    '''The Birmingham Method is an efficient, analytical technique for tracking
    tracers using the LoRs from PEPT data.

    Two main methods are provided: `fit_sample` for tracking a single numpy
    array of LoRs (i.e. a single sample) and `fit` which tracks all the samples
    encapsulated in a `pept.LineData` class *in parallel*.

    For the given `sample` of LoRs (a numpy.ndarray), this function minimises
    the distance between all of the LoRs, rejecting a fraction of lines that
    lie furthest away from the calculated distance. The process is repeated
    iteratively until a specified fraction (`fopt`) of the original subset of
    LORs remains.

    This class is a wrapper around the `birmingham_method` subroutine
    (implemented in C), providing tools for asynchronously tracking samples of
    LoRs. It can return `PointData` classes which can be easily manipulated and
    visualised.

    Attributes
    ----------
    fopt : float
        Floating-point number between 0 and 1, representing the target fraction
        of LoRs in a sample used to locate a tracer.

    get_used : bool, default False
        If True, attach an attribute ``._lines`` to the output PointData
        containing the sample of LoRs used (+ a column `used`).

    Examples
    --------
    A typical workflow would involve reading LoRs from a file, instantiating a
    `BirminghamMethod` class, tracking the tracer locations from the LoRs, and
    plotting them.

    >>> import pept
    >>> from pept.tracking.birmingham_method import BirminghamMethod

    >>> lors = pept.LineData(...)   # set sample_size and overlap appropriately
    >>> bham = BirminghamMethod()
    >>> locations = bham.fit(lors)  # this is a `pept.PointData` instance

    >>> grapher = PlotlyGrapher()
    >>> grapher.add_points(locations)
    >>> grapher.show()

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    pept.scanners.ParallelScreens : Initialise a `pept.LineData` instance from
                                    parallel screens PEPT detectors.
    '''

    def __init__(self, fopt = 0.5, get_used = False):
        '''`BirminghamMethod` class constructor.

        fopt : float, default 0.5
            Float number between 0 and 1, representing the fraction of
            remaining LORs in a sample used to locate the particle.

        verbose : bool, default False
            Print extra information when initialising this class.
        '''

        # Use @fopt.setter (below) to do the relevant type-checking when
        # setting fopt (self._fopt is the internal attribute, that we only
        # access through the getter and setter of the self.fopt property).
        self.fopt = float(fopt)
        self.get_used = bool(get_used)


    @beartype
    def fit_sample(self, sample):
        '''Use the Birmingham method to track a tracer location from a numpy
        array (i.e. one sample) of LoRs.

        For the given `sample` of LoRs (a numpy.ndarray), this function
        minimises the distance between all of the LoRs, rejecting a fraction of
        lines that lie furthest away from the calculated distance. The process
        is repeated iteratively until a specified fraction (`fopt`) of the
        original subset of LORs remains.

        Parameters
        ----------
        sample : (N, M>=7) numpy.ndarray
            The sample of LORs that will be clustered. Each LoR is expressed as
            a timestamps and a line defined by two points; the data columns are
            then `[time, x1, y1, z1, x2, y2, z2, extra...]`.

        get_used : bool, default False
            If `True`, the function will also return a boolean mask of the LoRs
            used to compute the tracer location - that is, a vector of the same
            length as `sample`, containing 1 for the rows that were used, and 0
            otherwise.

        as_array : bool, default True
            If set to True, the tracked locations are returned as numpy arrays.
            If set to False, they are returned inside an instance of
            `pept.PointData` for ease of iteration and plotting.

        verbose : bool, default False
            Provide extra information when tracking a location: time the
            operation and show a progress bar.

        Returns
        -------
        locations : numpy.ndarray or pept.PointData
            The tracked locations found.

        used : numpy.ndarray, optional
            If `get_used` is true, then also return a boolean mask of the LoRs
            used to compute the tracer location - that is, a vector of the same
            length as `sample`, containing 1 for the rows that were used, and 0
            otherwise.
            [ Used for multi-particle tracking, not implemented yet]

        Raises
        ------
        ValueError
            If `sample` is not a numpy array of shape (N, M), where M >= 7.
        '''

        if not isinstance(sample, pept.LineData):
            sample = pept.LineData(sample)

        locations, used = birmingham_method(sample.lines, self.fopt)

        # Propagate any LineData attributes besides `columns`
        attrs = sample.extra_attrs()

        locations = pept.PointData(
            [locations],
            columns = ["t", "x", "y", "z", "error"],
            **attrs,
        )

        # If `get_used`, also attach a `._lines` attribute with the lines used
        if self.get_used:
            locations.attrs["_lines"] = sample.copy(
                data = np.c_[sample.lines, used],
                columns = sample.columns + ["used"],
            )

        return locations
