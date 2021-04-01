#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
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

# File   : adac_forte.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.04.2021


import  os
import  time
import  textwrap

from    pept            import  LineData

from    .extensions     import  convert_adac_forte




class ADACForte(LineData):
    '''A subclass of `LineData` that initialises PEPT lines of response (LoRs)
    from the ADAC Forte parallel screen detector list mode *binary* format.

    Inherits all properties and methods from the general `LineData` class, but
    reads in LoRs from ADAC binary files (usual extension ".da01").

    Attributes
    ----------
    sample_size, overlap, number_of_lines, etc.: inherited from `pept.LineData`
        All attributes and methods from the parent class `pept.LineData` are
        available after instantiation. Check its documentation for more
        information.

    Methods
    -------
    to_csv, lines_trace, etc. : inherited from `pept.LineData`
        All attributes and methods from the parent class `pept.LineData` are
        available after instantiation. Check its documentation for more
        information.

    Examples
    --------
    Initialise a `ParallelScreens` array for three LoRs on a parallel screens
    PEPT scanner (i.e. each line is defined by **two** points each) with a
    head separation of 500 mm:

    >>> lors = pept.scanners.ADACForte("binary_data_adac.da01")
    >>> Initialising the PEPT data took 0.00038814544677734375 seconds.

    >>> lors
    >>> Class instance that inherits from `pept.LineData`.
    >>> Type:
    >>> <class 'pept.scanners.parallel_screens.adac_forte.ADACForte'>
    >>>
    >>> Attributes
    >>> ----------
    >>> number_of_lines =   3
    >>>
    >>> sample_size =       0
    >>> overlap =           0
    >>> number_of_samples = 1
    >>>
    >>> lines =
    >>> [[  2. 100. 150.   0. 200. 250. 500.]
    >>>  [  4. 350. 250.   0. 100. 150. 500.]
    >>>  [  6. 450. 350.   0. 250. 200. 500.]]
    >>>
    >>> Particular Cases
    >>> ----------------
    >>> > If sample_size == 0, all `lines` are returned as a single sample.
    >>> > If overlap >= sample_size, an error is raised.
    >>> > If overlap < 0, lines are skipped between samples.

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.utilities.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    '''

    def __init__(
        self,
        filepath,
        sample_size = 0,
        overlap = 0,
        verbose = True
    ):
        '''ParallelScreens class constructor.

        Parameters
        ----------
        filepath_or_array : str
            The path to a ADAC Forte-generated binary file from which the LoRs
            will be read into the `LineData` format.

        sample_size : int, default 0
            An `int` that defines the number of lines that should be returned
            when iterating over `lines`. A `sample_size` of 0 yields all the
            data as one single sample. A good starting value would be 200 times
            the maximum number of tracers that would be tracked.

        overlap : int, default 0
            An `int` that defines the overlap between two consecutive samples
            that are returned when iterating over `lines`. An overlap of 0
            implies consecutive samples, while an overlap of
            (`sample_size` - 1) means incrementing the samples by one. A
            negative overlap means skipping values between samples. An error is
            raised if `overlap` is larger than or equal to `sample_size`.

        verbose : bool, default True
            An option that enables printing the time taken for the
            initialisation of an instance of the class. Useful when reading
            large files (10gb files for PEPT data is not unheard of).

        Raises
        ------
        FileNotFoundError
            If the input `filepath` does not exist.

        ValueError
            If `overlap` >= `sample_size`. Overlap has to be smaller than
            `sample_size`. Note that it can also be negative.
        '''

        if verbose:
            start = time.time()

        if not os.path.isfile(filepath):
            raise FileNotFoundError(textwrap.fill((
                f"The input file path {filepath} does not exist!"
            )))

        lines = convert_adac_forte(filepath)

        # Call the constructor of the superclass `LineData` to initialise all
        # the inner parameters of the class (_index, etc.)
        LineData.__init__(
            self,
            lines,
            sample_size = sample_size,
            overlap = overlap,
            verbose = False
        )

        if verbose:
            end = time.time()
            print(f"Initialising the PEPT data took {end - start} seconds.\n")
