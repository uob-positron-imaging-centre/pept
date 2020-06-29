#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
#
#    Copyright (C) 2020 Sam Manger
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


import  os
import  time
import  textwrap
import  warnings
from    concurrent.futures              import  ThreadPoolExecutor

import  numpy                           as      np
from    tqdm                            import  tqdm

import  pept

from    .extensions.birmingham_method   import  birmingham_method


class BirminghamMethod:
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

    Methods
    -------
    fit_sample(sample, get_used = False, as_array = True, verbose = False)
        Use the Birmingham method to track a tracer location from a numpy
        array (i.e. one sample) of LoRs.
    fit(line_data, max_error = 10, get_used = False, max_workers = None,\
        verbose = True)
        Fit lines of response (an instance of 'LineData') and return the
        tracked locations and (optionally) the LoRs that were used.

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

    def __init__(self, fopt = 0.5, verbose = False):
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
        self.fopt = fopt

        if verbose:
            print("Initialised BirminghamMethod.")


    @property
    def fopt(self):
        '''The fraction of LORs used to locate a particle.

        fopt : float
            Float number between 0 and 1, representing the fraction of
            remaining LORs in a sample used to locate the particle.
        '''

        return self._fopt


    @fopt.setter
    def fopt(self, new_fopt):
        '''The fraction of LORs used to locate a particle

        Parameters
        ----------
        new_fopt : float
            Float number between 0 and 1, representing the fraction of
            remaining LORs in a sample used to locate the particle.

        Raises
        ------
        ValueError
            If 'new_fopt' is less than 0 or greater than 1.
        '''

        new_fopt = float(new_fopt)
        if new_fopt > 1 or new_fopt <= 0:
            raise ValueError(textwrap.fill(
                "[ERROR]: fopt should be set between 0 and 1. Received "
                f"{new_fopts}."
            ))

        self._fopt = new_fopt


    # Use the standardised function names `fit_sample` (for one numpy array)
    # and `fit` (for PointData or LineData).
    def fit_sample(
        self,
        sample,
        get_used = False,
        as_array = True,
        verbose = False
    ):
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

        if verbose:
            start = time.time()

        # Type-check input parameters.
        # sample cols: [time, x1, y1, z1, x2, y2, z2, etc.]
        sample = np.asarray(sample, dtype = float, order = "C")

        if sample.ndim != 2 or sample.shape[1] < 7:
            raise ValueError(textwrap.fill(
                "[ERROR]: `sample` should have two dimensions (M, N), where "
                f"N >= 7. Received {sample.shape}."
            ))

        locations, used = birmingham_method(sample, self._fopt)

        if not as_array:
            locations = pept.PointData(
                locations, sample_size = 0, overlap = 0, verbose = False
            )

        if verbose:
            end = time.time()
            print((
                "Tracking one location with %i LORs took %.3f seconds" %
                (sample.shape[0], end - start)
            ))

        if get_used:
            return locations, used

        return locations


    # Use the standardised function names `fit_sample` (for one numpy array)
    # and `fit` (for PointData or LineData).
    def fit(
        self,
        line_data,
        max_error = 10,
        get_used = False,
        max_workers = None,
        verbose = True
    ):
        '''Fit lines of response (an instance of 'LineData') and return the
        tracked locations and (optionally) the LoRs that were used.

        This is a convenience function that asynchronously iterates through the
        samples in a `LineData`, finding the tracer locations. For more
        fine-grained control over the tracking, the `fit_sample` method can be
        used for individual samples.

        Parameters
        ----------
        line_data : an instance of `pept.LineData`
            The samples of lines of reponse (LoRs) that will be used for
            locating the tracer. Be careful to set the appropriate
            `sample_size` and `overlap` for good results. If the `sample_size`
            is too low, the tracer might not be found; if it is too high,
            temporal resolution is decreased. If the `overlap` is too small,
            the tracked points might be very "sparse".
        max_error : float, default = 10
            The maximum error allowed to return a 'valid' tracked location. All
            tracer locations with an error larger than `max_error` will be
            discarded.
        get_used : bool, default False
            If `True`, the function will also return a list of boolean masks of
            the LoRs used to compute the tracer location for each sample - that
            is, a vector of the same length as `sample`, containing 1 for the
            rows that were used, and 0 otherwise.
        max_workers : int, optional
            The maximum number of threads that will be used for asynchronously
            clustering the samples in `cutpoints`. If unset (`None`), the
            number of threads available on the machine (as returned by
            `os.cpu_count()`) will be used.
        verbose : bool, default True
            Provide extra information when tracking: time the operation and
            show a progress bar.

        Returns
        -------
        locations : pept.PointData
            The tracer locations found.
        used : list of numpy.ndarray
            A list of boolean masks of the LoRs used to compute the tracer
            location for each corresponding sample in `line_data` - that is, a
            vector of the same length as a sample, containing 1 for the rows
            that were used, and 0 otherwise.

        Raises
        ------
        TypeError
            If `line_data` is not an instance of `pept.LineData`.
        '''

        if verbose:
            start = time.time()

        if not isinstance(line_data, pept.LineData):
            raise TypeError(textwrap.fill(
                "[ERROR]: `line_data` should be an instance of `pept.LineData`"
                f" (or any subclass thereof). Received {type(line_data)}."
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
                    ), replace_whitespace=False),
                RuntimeWarning
            )

        get_used = bool(get_used)

        # Using ThreadPoolExecutor, asynchronously collect the locations from
        # every sample in a list of arrays. This is more efficient than using
        # ProcessPoolExecutor (or joblib) because birmingham_method is a Cython
        # function that releases the GIL for most of its computation.
        # If verbose, show progress bar using tqdm.
        if max_workers is None:
            max_workers = os.cpu_count()

        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            futures = []
            for sample in line_data:
                futures.append(
                    executor.submit(
                        birmingham_method,
                        sample,
                        self._fopt
                    )
                )

            if verbose:
                futures = tqdm(futures)

            data_list = [f.result() for f in futures]

        # Access the data_list output as list comprehensions
        # data_list is a list of tuples, in which the first element is an
        # array of the `location`, and the second element is `used`, a
        # boolean mask representing the used LoRs.
        locations = [r[0] for r in data_list if len(r[0]) != 0]
        used = [r[1] for r in data_list if len(r[1]) != 0]

        # Remove LoRs with error above max_error
        locations = np.vstack(locations)
        locations = np.delete(
            locations,
            np.argwhere(locations[:, 4] > max_error),
            axis = 0
        )

        if len(locations) != 0:
            locations = pept.PointData(
                locations,
                sample_size = 0,
                overlap = 0,
                verbose = False
            )

        if verbose:
            end = time.time()
            print("\nTracking locations took {} seconds\n".format(end - start))

        if get_used:
            # `used is a list of the `used` arrays for the corresponding sample
            # in `line_data`.
            return locations, used

        return locations


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"fopt = {self.fopt}"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPL
        docstr = (
            "Class instance that inherits from `BirminghamMethod`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n"
            "----------\n"
            f"{self.__str__()}\n"
        )

        return docstr


