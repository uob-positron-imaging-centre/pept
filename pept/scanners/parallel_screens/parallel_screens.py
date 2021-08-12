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
#    Copyright (C) 2021 the pept developers
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

# File   : parallel_screens.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.08.2019


import  time
import  numpy           as      np

from    pept            import  LineData
from    pept.utilities  import  read_csv




def parallel_screens(
    filepath_or_array,
    screen_separation,
    sample_size = 0,
    overlap = 0,
    verbose = True,
    **kwargs,
):
    '''Initialise PEPT LoRs for parallel screens PET/PEPT detectors from an
    input CSV file or array.

    **The expected data columns in the file are `[time, x1, y1, x2, y2]`**.
    This is automatically transformed into the standard `Lines` format with
    columns being `[time, x1, y1, z1, x2, y2, z2]`, where `z1 = 0` and
    `z2 = screen_separation`.

    `ParallelScreens` can be initialised with a predefined numpy array of LoRs
    or read data from a `.csv`.

    Parameters
    ----------
    filepath_or_array : [str, pathlib.Path, IO] or numpy.ndarray (N, 5)
        A path to a file to be read from or an array for initialisation. A
        path is a string with the (absolute or relative) path to the data
        file or a URL from which the PEPT data will be read. It should
        include the full file name, along with its extension (.csv, .a01,
        etc.).

    screen_separation : float
        The separation (in *mm*) between the two PEPT screens corresponding
        to the `z` coordinate of the second point defining each line. The
        attribute `lines`, with columns
        `[time, x1, y1, z1, x2, y2, z2]`, will have `z1 = 0` and
        `z2 = screen_separation`.

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

    kwargs : other keyword arguments
        Other keyword arguments to be passed to `pept.read_csv`, e.g.
        "skiprows" or "max_rows". See the `pept.read_csv` documentation for
        other arguments.

    Returns
    -------
    LineData
        The initialised LoRs.

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size`. Overlap has to be smaller than
        `sample_size`. Note that it can also be negative.

    ValueError
        If the data file does not have the (N, M >= 5) shape.


    Examples
    --------
    Initialise a `LineData` array for three LoRs on a parallel screens
    PEPT scanner (i.e. each line is defined by **two** points each) with a
    head separation of 500 mm:

    >>> lors_raw = np.array([
    >>>     [2, 100, 150, 200, 250],
    >>>     [4, 350, 250, 100, 150],
    >>>     [6, 450, 350, 250, 200]
    >>> ])

    >>> screen_separation = 500
    >>> lors = pept.scanners.parallel_screens(lors_raw, screen_separation)
    Initialised PEPT data in 0.001 s.

    >>> lors
    LineData
    --------
    sample_size = 0
    overlap =     0
    samples =     1
    lines =
      [[  2. 100. 150.   0. 200. 250. 500.]
       [  4. 350. 250.   0. 100. 150. 500.]
       [  6. 450. 350.   0. 250. 200. 500.]]
    lines.shape = (3, 7)
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']


    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    '''

    if verbose:
        start = time.time()

    # Check wheter input is a valid `pandas.read_csv` filepath or a numpy
    # array in an "Easier To Ask Forgiveness Than Permission" way.
    try:
        # Try to read the LoR data from `filepath_or_array`.
        # Check if an error is raised when reading file lines using
        lines = read_csv(filepath_or_array, **kwargs)
    except ValueError:
        # Seems like it is an array!
        lines = np.asarray(filepath_or_array, order = "C", dtype = float)

    # Add Z1 and Z2 columns => [time, X1, Y1, Z1, X2, Y2, Z2]
    # Z1 = 0
    lines = np.insert(lines, 3, 0.0, axis = 1)

    # Z2 = `separation`
    lines = np.insert(lines, 6, screen_separation, axis = 1)

    if verbose:
        end = time.time()
        print(f"Initialised PEPT data in {end - start:3.3f} s.\n")

    return LineData(lines, sample_size = sample_size, overlap = overlap)
