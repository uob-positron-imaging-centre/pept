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


# File   : cutpoints.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 13.04.2020


import numpy as np
import pept


def find_cutpoints(
    sample_lines,
    max_distance,
    cutoffs = None,
    append_indices = False
):
    '''Find the cutpoints from a sample / array of LoRs.

    A cutpoint is the point in 3D space that minimises the distance between any
    two lines. For any two non-parallel 3D lines, this point corresponds to the
    midpoint of the unique segment that is perpendicular to both lines.

    This function considers every pair of lines in `sample_lines` and returns
    all the cutpoints that satisfy the following conditions:

    1. The distance between the two lines is smaller than `max_distance`.
    2. The cutpoint is within the `cutoffs`.

    Parameters
    ----------
    sample_lines : (N, M >= 7) numpy.ndarray
        A sample of LoRs, where each row is `[time, x1, y1, z1, x2, y2, z2]`,
        such that every line is defined by the points `[x1, y1, z1]` and
        `[x2, y2, z2]`.
    max_distance : float
        The maximum distance between any two lines for their cutpoint to be
        considered. A good starting value would be 0.1 mm for small tracers
        and/or clean data, or 0.2 mm for larger tracers and/or noisy data.
    cutoffs : list, optional
        The cutoffs for each dimension, formatted as `[x_min, x_max,
        y_min, y_max, z_min, z_max]`. If it is `None`, they are computed
        automatically by calling `get_cutoffs`. The default is `None`.
    append_indices : bool, optional
        If set to `True`, the indices of the individual LoRs that were used
        to compute each cutpoint are also appended to the returned array.
        Default is `False`.

    Returns
    -------
    cutpoints : (M, 4) or (M, 6) numpy.ndarray
        A numpy array of the calculated cutpoints. If `append_indices` is
        `False`, then the columns are [time, x, y, z]. If `append_indices` is
        `True`, then the columns are [time, x, y, z, i, j], where `i` and `j`
        are the LoR indices from `sample_lines` that were used to compute the
        weighted cutpoints. The time is the average between the timestamps of
        the two LoRs that were used to compute the cutpoint. The first column
        (for time) is sorted.

    Raises
    ------
    ValueError
        If `sample_lines` is not a numpy array with shape (N, M >= 7).
    ValueError
        If `cutoffs` is not a one-dimensional array with values
        `[min_x, max_x, min_y, max_y, min_z, max_z]`

    See Also
    --------
    pept.tracking.peptml.Cutpoints : Compute cutpoints from `pept.LineData`.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    '''

    if not isinstance(sample_lines, pept.LineData):
        sample_lines = pept.LineData(sample_lines)

    lines = sample_lines.lines

    lines = np.asarray(lines, order = 'C', dtype = float)
    max_distance = float(max_distance)

    # If cutoffs were not defined, automatically compute them
    if cutoffs is None:
        cutoffs = get_cutoffs(lines)
    else:
        cutoffs = np.asarray(cutoffs, order = 'C', dtype = float)
        if cutoffs.ndim != 1 or len(cutoffs) != 6:
            raise ValueError((
                "\n[ERROR]: cutoffs should be a one-dimensional array with "
                "values [min_x, max_x, min_y, max_y, min_z, max_z]. Received "
                f"{cutoffs}.\n"
            ))

    sample_cutpoints = pept.utilities.find_cutpoints(
        lines,
        max_distance,
        cutoffs,
        append_indices = append_indices
    )

    columns = ["t", "x", "y", "z"]
    if append_indices:
        columns += ["line_index1", "line_index2"]

    points = pept.PointData(sample_cutpoints, columns = columns)

    # Add optional metadata to the points; because they have an underscore,
    # they won't be propagated when new objects are constructed
    points._max_distance = max_distance
    points._cutoffs = cutoffs
    if append_indices:
        points._lines = sample_lines

    return points


def get_cutoffs(sample):
    '''Compute the cutoffs from a sample of LoR data.

    It computes the cutoffs from the minimum and maximum values of the LoRs in
    `sample` in each dimension (e.g. the x-dimension is defined by data in
    columns 1 and 4).

    Parameters
    ----------
    sample : (N, M >= 7) numpy.ndarray
        A sample of LoRs, where each row is `[time, x1, y1, z1, x2, y2, z2]`,
        such that every line is defined by the points `[x1, y1, z1]` and
        `[x2, y2, z2]`.

    Returns
    -------
    cutoffs : (6,) numpy.ndarray
        The computed cutoffs for each dimension, formatted as
        `[x_min, x_max, y_min, y_max, z_min, z_max]`.

    Raises
    ------
    ValueError
        If `sample` is not a numpy array with shape (N, M >= 7).

    See Also
    --------
    pept.tracking.peptml.Cutpoints : Compute cutpoints from `pept.LineData`.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    '''

    # Check sample has shape (N, M >= 7)
    if sample.ndim != 2 or sample.shape[1] < 7:
        raise ValueError((
            "\n[ERROR]: `sample_lines` should have dimensions (M, N), "
            f" where N >= 7. Received {sample.shape}.\n"
        ))

    # Compute cutoffs for cutpoints as the (min, max) values of the lines
    # Minimum value of the two points that define a line
    min_x = min(sample[:, 1].min(),
                sample[:, 4].min())
    # Maximum value of the two points that define a line
    max_x = max(sample[:, 1].max(),
                sample[:, 4].max())

    # Minimum value of the two points that define a line
    min_y = min(sample[:, 2].min(),
                sample[:, 5].min())
    # Maximum value of the two points that define a line
    max_y = max(sample[:, 2].max(),
                sample[:, 5].max())

    # Minimum value of the two points that define a line
    min_z = min(sample[:, 3].min(),
                sample[:, 6].min())
    # Maximum value of the two points that define a line
    max_z = max(sample[:, 3].max(),
                sample[:, 6].max())

    cutoffs = np.array([min_x, max_x, min_y, max_y, min_z, max_z],
                       dtype = float)
    return cutoffs




class Cutpoints(pept.base.LineDataFilter):
    '''Transform LoRs (a `pept.LineData` instance) into *cutpoints* (a
    `pept.PointData` instance) for clustering, in parallel.

    Under typical usage, the `Cutpoints` class is initialised with a
    `pept.LineData` instance, automatically calculating the cutpoints from the
    samples of lines. The `Cutpoints` class inherits from `pept.PointData`,
    such that once the cutpoints have been computed, all the methods from the
    parent class `pept.PointData` can be used on them (such as visualisation
    functionality).

    For more control over the operations, `pept.tracking.peptml.find_cutpoints`
    can be used - it receives a generic numpy array of LoRs (one 'sample') and
    returns a numpy array of cutpoints.

    Attributes
    ----------
    max_distance : float
        The maximum distance between any two lines for their cutpoint to be
        considered. A good starting value would be 0.1 mm for small tracers
        and/or clean data, or 0.2 mm for larger tracers and/or noisy data.

    cutoffs : list-like of length 6
        A list (or equivalent) of the cutoff distances for every axis,
        formatted as `[x_min, x_max, y_min, y_max, z_min, z_max]`. Only the
        cutpoints which fall within these cutoff distances are considered. The
        default is None, in which case they are automatically computed using
        `pept.tracking.peptml.get_cutoffs`.

    Examples
    --------
    Compute the cutpoints for a `LineData` instance between lines that are less
    than 0.1 apart:

    >>> line_data = pept.LineData(example_data)
    >>> cutpts = peptml.Cutpoints(0.1).fit(line_data)

    Compute the cutpoints for a single sample:

    >>> sample = line_data[0]
    >>> cutpts_sample = peptml.Cutpoints(0.1).fit_sample(sample)

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.tracking.HDBSCAN : Efficient, parallel HDBSCAN-based clustering of
                            (cut)points.
    pept.read_csv : Fast CSV file reading into numpy arrays.
    '''

    def __init__(
        self,
        max_distance,
        cutoffs = None,
        append_indices = False,
    ):
        '''Cutpoints class constructor.

        Parameters
        ----------
        line_data : instance of pept.LineData
            The LoRs for which the cutpoints will be computed. It must be an
            instance of `pept.LineData`.

        max_distance : float
            The maximum distance between any two lines for their cutpoint to be
            considered. A good starting value would be 0.1 mm for small tracers
            and/or clean data, or 0.5 mm for larger tracers and/or noisy data.

        cutoffs : list-like of length 6, optional
            A list (or equivalent) of the cutoff distances for every axis,
            formatted as `[x_min, x_max, y_min, y_max, z_min, z_max]`. Only the
            cutpoints which fall within these cutoff distances are considered.
            The default is None, in which case they are automatically computed
            using `pept.tracking.peptml.get_cutoffs`.

        append_indices : bool, default False
            If set to `True`, the indices of the individual LoRs that were used
            to compute each cutpoint are also appended to the returned array.

        Raises
        ------
        ValueError
            If `cutoffs` is not a one-dimensional array with values formatted
            as `[min_x, max_x, min_y, max_y, min_z, max_z]`.
        '''

        # Setting class attributes. The ones below call setters which do type
        # checking
        self.cutoffs = cutoffs
        self.append_indices = append_indices
        self.max_distance = max_distance


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

        sample_cutpoints = pept.utilities.find_cutpoints(
            sample_lines.lines,
            self.max_distance,
            cutoffs,
            append_indices = self.append_indices,
        )

        columns = ["t", "x", "y", "z"]
        if self.append_indices:
            columns += ["line_index1", "line_index2"]

        points = pept.PointData(sample_cutpoints, columns = columns)

        # Add optional metadata to the points; because they have an underscore,
        # they won't be propagated when new objects are constructed
        points.attrs["_max_distance"] = self.max_distance
        points.attrs["_cutoffs"] = self.cutoffs

        if self.append_indices:
            points.attrs["_lines"] = sample_lines

        return points
