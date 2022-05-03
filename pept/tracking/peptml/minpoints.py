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


# File   : minpoints.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.10.2020


import  numpy               as      np

import  pept
from    .cutpoints          import  get_cutoffs


def find_minpoints(
    sample_lines,
    num_lines,
    max_distance,
    cutoffs = None,
    append_indices = False
):
    '''Compute the minimum distance points (MDPs) from all combinations of
    `num_lines` lines given in an array of lines `sample_lines`.

    Given a sample of lines, this functions computes the minimum distance
    points (MDPs) for every possible combination of `num_lines` lines. The
    returned numpy array contains all MDPs that satisfy the following:

    1. Are within the `cutoffs`.
    2. Are closer to all the constituent LoRs than `max_distance`.

    Parameters
    ----------
    sample_lines : (M, N) numpy.ndarray
        A 2D array of lines, where each line is defined by two points such that
        every row is formatted as `[t, x1, y1, z1, x2, y2, z2, etc.]`. It
        *must* have at least 2 lines and the combination size `num_lines`
        *must* be smaller or equal to the number of lines. Put differently:
        2 <= num_lines <= len(sample_lines).

    num_lines : int
        The number of lines in each combination of LoRs used to compute the
        MDP. This function considers every combination of `numlines` from the
        input `sample_lines`. It must be smaller or equal to the number of
        input lines `sample_lines`.

    max_distance : float
        The maximum allowed distance between an MDP and its constituent lines.
        If any distance from the MDP to one of its lines is larger than
        `max_distance`, the MDP is thrown away.

    cutoffs : (6,) numpy.ndarray, optional
        An array of spatial cutoff coordinates with *exactly 6 elements* as
        [x_min, x_max, y_min, y_max, z_min, z_max]. If any MDP lies outside
        this region, it is thrown away. If it is `None`, they are computed
        automatically by calling `get_cutoffs`. The default is `None`.

    append_indices : bool, default False
        A boolean specifying whether to include the indices of the lines used
        to compute each MDP. If `False`, the output array will only contain the
        [time, x, y, z] of the MDPs. If `True`, the output array will have
        extra columns [time, x, y, z, line_idx(1), ..., line_idx(n)] where
        n = `num_lines`.

    Returns
    -------
    minpoints : (M, N) numpy.ndarray
        A 2D array of `float`s containing the time and coordinates of the MDPs
        [time, x, y, z]. The time is computed as the average of the constituent
        lines. If `append_indices` is `True`, then `num_lines` indices of the
        constituent lines are appended as extra columns:
        [time, x, y, z, line_idx1, line_idx2, ..]. The first column (for time)
        is sorted.

    Raises
    ------
    ValueError
        If `sample_lines` is not a numpy array with shape (N, M >= 7).

    ValueError
        If 2 <= num_lines <= len(sample_lines) is not satisfied.

    ValueError
        If `cutoffs` is not a one-dimensional array with values
        `[min_x, max_x, min_y, max_y, min_z, max_z]`

    See Also
    --------
    pept.tracking.peptml.Minpoints : Compute minpoints from `pept.LineData`.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    '''

    if not isinstance(sample_lines, pept.LineData):
        sample_lines = pept.LineData(sample_lines)

    lines = sample_lines.lines

    lines = np.asarray(lines, order = 'C', dtype = float)

    num_lines = int(num_lines)
    max_distance = float(max_distance)

    if cutoffs is None:
        cutoffs = get_cutoffs(sample_lines)
    else:
        cutoffs = np.asarray(cutoffs, order = 'C', dtype = float)
        if cutoffs.ndim != 1 or len(cutoffs) != 6:
            raise ValueError((
                "\n[ERROR]: cutoffs should be a one-dimensional array with "
                "values [min_x, max_x, min_y, max_y, min_z, max_z]. Received "
                f"{cutoffs}.\n"
            ))

    sample_minpoints = pept.utilities.find_minpoints(
        lines,
        num_lines,
        max_distance,
        cutoffs,
        append_indices = append_indices
    )

    columns = ["t", "x", "y", "z"]
    if append_indices:
        columns += [f"line_index{i + 1}" for i in range(num_lines)]

    points = pept.PointData(sample_minpoints, columns = columns)

    # Add optional metadata to the points; because they have an underscore,
    # they won't be propagated when new objects are constructed
    points._max_distance = max_distance
    points._cutoffs = cutoffs
    points._num_lines = num_lines

    if append_indices:
        points._lines = sample_lines

    return points




class Minpoints(pept.base.LineDataFilter):
    '''Transform LoRs (a `pept.LineData` instance) into *minpoints* (a
    `pept.PointData` instance) for clustering, in parallel.

    Given a sample of lines, the minpoints are the minimum distance points
    (MDPs) for every possible combination of `num_lines` lines that satisfy
    the following conditions:

    1. Are within the `cutoffs`.
    2. Are closer to all the constituent LoRs than `max_distance`.

    Under typical usage, the `Minpoints` class is initialised with a
    `pept.LineData` instance, automatically calculating the minpoints from the
    samples of lines. The `Minpoints` class inherits from `pept.PointData`,
    such that once the cutpoints have been computed, all the methods from the
    parent class `pept.PointData` can be used on them (such as visualisation
    functionality).

    For more control over the operations, `pept.tracking.peptml.find_minpoints`
    can be used - it receives a generic numpy array of LoRs (one 'sample') and
    returns a numpy array of cutpoints.

    Attributes
    ----------
    line_data : instance of pept.LineData
        The LoRs for which the cutpoints will be computed. It must be an
        instance of `pept.LineData`.

    num_lines : int
        The number of lines in each combination of LoRs used to compute the
        MDP. This function considers every combination of `num_lines` from the
        input `sample_lines`. It must be smaller or equal to the number of
        input lines `sample_lines`.

    max_distance : float
        The maximum allowed distance between an MDP and its constituent lines.
        If any distance from the MDP to one of its lines is larger than
        `max_distance`, the MDP is thrown away. A good starting value would be
        0.1 mm for small tracers and/or clean data, or 0.2 mm for larger
        tracers and/or noisy data.

    cutoffs : list-like of length 6
        A list (or equivalent) of the cutoff distances for every axis,
        formatted as `[x_min, x_max, y_min, y_max, z_min, z_max]`. Only the
        minpoints which fall within these cutoff distances are considered. The
        default is None, in which case they are automatically computed using
        `pept.tracking.peptml.get_cutoffs`.

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.tracking.peptml.HDBSCANClusterer : Efficient, parallel HDBSCAN-based
                                            clustering of cutpoints.
    pept.scanners.ParallelScreens : Read in and initialise a `pept.LineData`
                                    instance from parallel screens PET/PEPT
                                    detectors.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.

    Examples
    --------
    Compute the minpoints for a `LineData` instance for all triplets of lines
    that are less than 0.1 from those lines:

    >>> line_data = pept.LineData(example_data)
    >>> minpts = peptml.Minpoints(line_data, 3, 0.1)

    Compute the minpoints for a single sample:

    >>> sample = line_data[0]
    >>> cutpts_sample = peptml.find_minpoints(sample, 3, 0.1)
    '''

    def __init__(
        self,
        num_lines,
        max_distance,
        cutoffs = None,
        append_indices = False,
    ):
        '''Minpoints class constructor.

        Parameters
        ----------
        num_lines : int
            The number of lines in each combination of LoRs used to compute the
            MDP. This function considers every combination of `num_lines` from
            the input `sample_lines`. It must be smaller or equal to the number
            of input lines `sample_lines`.

        max_distance : float
            The maximum allowed distance between an MDP and its constituent
            lines. If any distance from the MDP to one of its lines is larger
            than `max_distance`, the MDP is thrown away. A good starting value
            would be 0.1 mm for small tracers and/or clean data, or 0.2 mm for
            larger tracers and/or noisy data.

        cutoffs : list-like of length 6, optional
            A list (or equivalent) of the cutoff distances for every axis,
            formatted as `[x_min, x_max, y_min, y_max, z_min, z_max]`. Only the
            minpoints which fall within these cutoff distances are considered.
            The default is None, in which case they are automatically computed
            using `pept.tracking.peptml.get_cutoffs`.

        append_indices : bool, default False
            If set to `True`, the indices of the individual LoRs that were used
            to compute each minpoint are also appended to the returned array.

        Raises
        ------
        TypeError
            If `line_data` is not an instance of `pept.LineData`.

        ValueError
            If 2 <= num_lines <= len(sample_lines) is not satisfied.

        ValueError
            If `cutoffs` is not a one-dimensional array with values formatted
            as `[min_x, max_x, min_y, max_y, min_z, max_z]`.
        '''

        # Setting class attributes. The ones below call setters which do type
        # checking
        self.num_lines = num_lines
        self.max_distance = max_distance
        self.cutoffs = cutoffs
        self.append_indices = append_indices


    @property
    def num_lines(self):
        return self._num_lines


    @num_lines.setter
    def num_lines(self, num_lines):
        self._num_lines = int(num_lines)


    @property
    def max_distance(self):
        return self._max_distance


    @max_distance.setter
    def max_distance(self, max_distance):
        self._max_distance = float(max_distance)


    @property
    def cutoffs(self):
        return self._cutoffs


    @cutoffs.setter
    def cutoffs(self, cutoffs):
        if cutoffs is not None:
            cutoffs = np.asarray(cutoffs, order = 'C', dtype = float)
            if cutoffs.ndim != 1 or len(cutoffs) != 6:
                raise ValueError((
                    "\n[ERROR]: cutoffs should be a one-dimensional array "
                    "with values [min_x, max_x, min_y, max_y, min_z, max_z]. "
                    f"Received {cutoffs}.\n"
                ))

            self._cutoffs = cutoffs
        else:
            self._cutoffs = None


    @property
    def append_indices(self):
        return self._append_indices


    @append_indices.setter
    def append_indices(self, append_indices):
        self._append_indices = bool(append_indices)


    def fit_sample(self, sample_lines):
        if not isinstance(sample_lines, pept.LineData):
            sample_lines = pept.LineData(sample_lines)

        # If cutoffs were not defined, automatically compute them
        if self.cutoffs is not None:
            cutoffs = self.cutoffs
        else:
            cutoffs = get_cutoffs(sample_lines.lines)

        # Only compute minpoints if there are at least num_lines LoRs
        if len(sample_lines.lines) >= self.num_lines:
            sample_minpoints = pept.utilities.find_minpoints(
                sample_lines.lines,
                self.num_lines,
                self.max_distance,
                cutoffs,
                append_indices = self.append_indices,
            )
        else:
            ncols = 4 + self.num_lines if self.append_indices else 4
            sample_minpoints = np.empty((0, ncols))

        # Column names
        columns = ["t", "x", "y", "z"]
        if self.append_indices:
            columns += [f"line_index{i + 1}" for i in range(self.num_lines)]

        # Encapsulate minpoints in a PointData
        points = pept.PointData(sample_minpoints, columns = columns)

        # Add optional metadata to the points; because they have an underscore,
        # they won't be propagated when new objects are constructed
        points.attrs["_num_lines"] = self.num_lines
        points.attrs["_max_distance"] = self.max_distance
        points.attrs["_cutoffs"] = cutoffs

        # If LoR indices were appended, also include the constituent LoRs
        if self.append_indices:
            points.attrs["_lines"] = sample_lines

        return points
