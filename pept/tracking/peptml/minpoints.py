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
#    Copyright (C) 2020 Andrei Leonard Nicusan
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


import  time
import  os
import  warnings
import  textwrap

import  numpy               as      np

from    joblib              import  Parallel, delayed
from    tqdm                import  tqdm

import  pept
from    pept.tracking       import  peptml


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
    sample_lines: (M, N) numpy.ndarray
        A 2D array of lines, where each line is defined by two points such that
        every row is formatted as `[t, x1, y1, z1, x2, y2, z2, etc.]`. It
        *must* have at least 2 lines and the combination size `num_lines`
        *must* be smaller or equal to the number of lines. Put differently:
        2 <= num_lines <= len(sample_lines).

    num_lines: int
        The number of lines in each combination of LoRs used to compute the
        MDP. This function considers every combination of `numlines` from the
        input `sample_lines`. It must be smaller or equal to the number of
        input lines `sample_lines`.

    max_distance: float
        The maximum allowed distance between an MDP and its constituent lines.
        If any distance from the MDP to one of its lines is larger than
        `max_distance`, the MDP is thrown away.

    cutoffs: (6,) numpy.ndarray, optional
        An array of spatial cutoff coordinates with *exactly 6 elements* as
        [x_min, x_max, y_min, y_max, z_min, z_max]. If any MDP lies outside
        this region, it is thrown away. If it is `None`, they are computed
        automatically by calling `get_cutoffs`. The default is `None`.

    append_indices: bool, default False
        A boolean specifying whether to include the indices of the lines used
        to compute each MDP. If `False`, the output array will only contain the
        [time, x, y, z] of the MDPs. If `True`, the output array will have
        extra columns [time, x, y, z, line_idx(1), ..., line_idx(n)] where
        n = `num_lines`.

    Returns
    -------
    minpoints: (M, N) numpy.ndarray
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

    sample_lines = np.asarray(sample_lines, order = 'C', dtype = float)
    num_lines = int(num_lines)
    max_distance = float(max_distance)

    # Check sample has shape (N, M >= 7)
    if sample_lines.ndim != 2 or sample_lines.shape[1] < 7:
        raise ValueError((
            "\n[ERROR]: `sample_lines` should have dimensions (M, N), "
            f" where N >= 7. Received {sample_lines.shape}.\n"
        ))

    if not 2 <= num_lines <= len(sample_lines):
        raise ValueError((
            "\n[ERROR]: The number of lines in a combination must be smaller "
            "than the number of lines in the input `sample_lines`:\n"
            "2 <= num_lines <= len(sample_lines)\n"
        ))

    if cutoffs is None:
        cutoffs = peptml.get_cutoffs(sample_lines)
    else:
        cutoffs = np.asarray(cutoffs, order = 'C', dtype = float)
        if cutoffs.ndim != 1 or len(cutoffs) != 6:
            raise ValueError((
                "\n[ERROR]: cutoffs should be a one-dimensional array with "
                "values [min_x, max_x, min_y, max_y, min_z, max_z]. Received "
                f"{cutoffs}.\n"
            ))

    sample_minpoints = pept.utilities.find_minpoints(
        sample_lines,
        num_lines,
        max_distance,
        cutoffs,
        append_indices = append_indices
    )

    return sample_minpoints




class Minpoints(pept.PointData):
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

    num_lines: int
        The number of lines in each combination of LoRs used to compute the
        MDP. This function considers every combination of `num_lines` from the
        input `sample_lines`. It must be smaller or equal to the number of
        input lines `sample_lines`.

    max_distance: float
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

    sample_size, overlap, number_of_lines, etc. : inherited from pept.PointData
        Additional attributes and methods are inherited from the base class
        `PointData`. Check its documentation for more information.

    Methods
    -------
    find_minpoints(line_data, num_lines, max_distance, cutoffs = None,\
                   append_indices = False, max_workers = None, verbose = True)
        Compute the minpoints from the samples in a `LineData` instance.

    sample, to_csv, plot, etc. : inherited from pept.PointData
        Additional attributes and methods are inherited from the base class
        `PointData`. Check its documentation for more information.

    Notes
    -----
    Once instantiated with a `LineData`, the class computes the minpoints and
    *automatically sets the sample_size* to the average number of minpoints
    found per sample of LoRs.

    Examples
    --------
    Compute the minpoints for a `LineData` instance for all triplets of lines
    that are less than 0.1 from those lines:

    >>> line_data = pept.LineData(example_data)
    >>> minpts = peptml.Minpoints(line_data, 3, 0.1)

    Compute the minpoints for a single sample:

    >>> sample = line_data[0]
    >>> cutpts_sample = peptml.find_minpoints(sample, 3, 0.1)

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.tracking.peptml.HDBSCANClusterer : Efficient, parallel HDBSCAN-based
                                            clustering of cutpoints.
    pept.scanners.ParallelScreens : Read in and initialise a `pept.LineData`
                                    instance from parallel screens PET/PEPT
                                    detectors.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    '''

    def __init__(
        self,
        line_data,
        num_lines,
        max_distance,
        cutoffs = None,
        append_indices = False,
        max_workers = None,
        verbose = True
    ):
        '''Cutpoints class constructor.

        Parameters
        ----------
        line_data : instance of pept.LineData
            The LoRs for which the cutpoints will be computed. It must be an
            instance of `pept.LineData`.

        num_lines: int
            The number of lines in each combination of LoRs used to compute the
            MDP. This function considers every combination of `num_lines` from
            the input `sample_lines`. It must be smaller or equal to the number
            of input lines `sample_lines`.

        max_distance: float
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

        max_workers : int, optional
            The maximum number of threads that will be used for asynchronously
            computing the minpoints from the samples of LoRs in `line_data`.

        verbose : bool, default True
            Provide extra information when computing the cutpoints: time the
            operation and show a progress bar.

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

        # Find the cutpoints when instantiated. The method
        # also initialises the instance as a `PointData` subclass.
        self.find_minpoints(
            line_data,
            num_lines,
            max_distance,
            cutoffs = cutoffs,
            append_indices = append_indices,
            max_workers = max_workers,
            verbose = verbose
        )


    @property
    def line_data(self):
        return self._line_data


    @property
    def num_lines(self):
        return self._num_lines


    @property
    def max_distance(self):
        return self._max_distance


    @property
    def cutoffs(self):
        return self._cutoffs


    def find_minpoints(
        self,
        line_data,
        num_lines,
        max_distance,
        cutoffs = None,
        append_indices = False,
        max_workers = None,
        verbose = True
    ):
        '''Compute the cutpoints from the samples in a `LineData` instance.

        Parameters
        ----------
        line_data : instance of pept.LineData
            The LoRs for which the cutpoints will be computed. It must be an
            instance of `pept.LineData`.

        num_lines: int
            The number of lines in each combination of LoRs used to compute the
            MDP. This function considers every combination of `num_lines` from
            the input `sample_lines`. It must be smaller or equal to the number
            of input lines `sample_lines`.

        max_distance: float
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

        max_workers : int, optional
            The maximum number of threads that will be used for asynchronously
            computing the minpoints from the samples of LoRs in `line_data`.

        verbose : bool, default True
            Provide extra information when computing the cutpoints: time the
            operation and show a progress bar.

        Returns
        -------
        self : the PointData instance of cutpoints
            The computed cutpoints are stored in the `Cutpoints` class, as a
            subclass of `pept.PointData`.

        Raises
        ------
        TypeError
            If `line_data` is not an instance of `pept.LineData`.

        ValueError
            If 2 <= num_lines <= len(sample_lines) is not satisfied.

        ValueError
            If `cutoffs` is not a one-dimensional array with values formatted
            as `[min_x, max_x, min_y, max_y, min_z, max_z]`.

        Notes
        -----
        This method is automatically called when instantiating this class.
        '''

        if verbose:
            start = time.time()

        # Check line_data is an instance (or a subclass!) of pept.LineData
        if not isinstance(line_data, pept.LineData):
            raise TypeError((
                "\n[ERROR]: line_data should be an instance (or subclass) of "
                "`pept.LineData`.\n"
            ))

        if not 2 <= num_lines <= line_data.sample_size:
            raise ValueError((
                "\n[ERROR]: The number of lines in a combination must be "
                "smaller than the LineData sample_size:"
                "\n2 <= num_lines <= line_data.sample_size\n"
            ))

        # Users might forget to set the sample_size, leaving it to the default
        # value of 0; in that case, all lines are returned as a single sample -
        # that might not be the intended behaviour.
        if line_data.sample_size == 0:
            warnings.warn(
                textwrap.fill((
                    "\n[WARNING]: The `line_data.sample_size` was left to the "
                    "default value of 0, in which case all lines are returned "
                    "as a single sample. For a very large number of lines, "
                    "this might result in a long function execution time.\n"
                ), replace_whitespace = False),
                RuntimeWarning
            )

        self._line_data = line_data
        self._max_distance = float(max_distance)
        self._num_lines = int(num_lines)

        # If cutoffs were not supplied, compute them
        if cutoffs is None:
            cutoffs = peptml.get_cutoffs(line_data.lines)
        # Otherwise make sure they are a C-contiguous numpy array
        else:
            cutoffs = np.asarray(cutoffs, order = 'C', dtype = float)
            if cutoffs.ndim != 1 or len(cutoffs) != 6:
                raise ValueError((
                    "\n[ERROR]: cutoffs should be a one-dimensional array "
                    "with values [min_x, max_x, min_y, max_y, min_z, max_z]. "
                    f"Received {cutoffs}.\n"
                ))

        self._cutoffs = cutoffs

        # Using ThreadPoolExecutor, asynchronously collect the minpoints from
        # every sample in a list of arrays. This is more efficient than using
        # ProcessPoolExecutor because find_minpoints is a Cython function that
        # releases the GIL for most of its computation.
        # If verbose, show progress bar using tqdm.
        if max_workers is None:
            max_workers = os.cpu_count()

        '''
        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            futures = []
            for sample in line_data:
                futures.append(
                    executor.submit(
                        pept.utilities.find_minpoints,
                        sample,
                        self._num_lines,
                        self._max_distance,
                        self._cutoffs,
                        append_indices = append_indices
                    )
                )

            if verbose:
                futures = tqdm(futures)

            minpoints = [f.result() for f in futures]
        '''

        minpoints = Parallel(n_jobs = max_workers)(
            delayed(pept.utilities.find_minpoints)(
                sample,
                self._num_lines,
                self._max_distance,
                self._cutoffs,
                append_indices = append_indices
            ) for sample in tqdm(line_data)
        )


        # minpoints shape: (n, m, 4), where n is the number of samples, and
        # m is the number of minpoints in the sample
        number_of_samples = len(minpoints)
        minpoints = np.vstack(minpoints)
        number_of_minpoints = len(minpoints)

        # Average number of minpoints per sample
        minpoints_per_sample = int(number_of_minpoints / number_of_samples)

        pept.PointData.__init__(
            self,
            minpoints,
            sample_size = minpoints_per_sample,
            overlap = 0,
            verbose = False
        )

        if verbose:
            end = time.time()
            print(f"\nFinding the minpoints took {end - start} seconds.\n")

        return self


    def __repr__(self):
        # Called when writing the class on a REPL. Add another line to the
        # standard description given in the parent class, pept.PointData.
        docstr = pept.PointData.__repr__(self) + (
            "\n\nNotes\n-----\n"
            "Once instantiated with a `LineData`, the class computes the \n"
            "minpoints and *automatically sets the sample_size* to the \n"
            "average number of minpoints found per sample of LoRs."
        )

        return docstr
