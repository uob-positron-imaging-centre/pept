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
from    glob            import  glob

import  numpy           as      np
from    natsort         import  natsorted
from    pept            import  LineData

from    .extensions     import  convert_adac_forte




def adac_forte(
    filepath,
    sample_size = None,
    overlap = None,
    verbose = True,
):
    '''Initialise PEPT lines of response (LoRs) from a binary file outputted by
    the ADAC Forte parallel screen detector list mode (common file extension
    ".da01").

    Parameters
    ----------
    filepath : str
        The path to a ADAC Forte-generated binary file from which the LoRs
        will be read into the `LineData` format. If you have multiple files,
        use a wildcard (*) after their common substring to concatenate them,
        e.g. "DS1.da*" will add ["DS1.da01", "DS1.da02", "DS1.da02_02"].

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

    Returns
    -------
    LineData
        The initialised LoRs.

    Raises
    ------
    FileNotFoundError
        If the input `filepath` does not exist.

    ValueError
        If `overlap` >= `sample_size`. Overlap has to be smaller than
        `sample_size`. Note that it can also be negative.

    See Also
    --------
    pept.LineData : Encapsulate LoRs for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    pept.read_csv : Fast CSV file reading into numpy arrays.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.

    Examples
    --------
    Initialise a `ParallelScreens` array for three LoRs on a parallel screens
    PEPT scanner (i.e. each line is defined by **two** points each) with a
    head separation of 500 mm:

    >>> lors = pept.scanners.adac_forte("binary_data_adac.da01")
    Initialised the PEPT data in 0.011 s.

    >>> lors
    LineData
    --------
    sample_size = 0
    overlap =     0
    samples =     1
    lines =
      [[0.00000000e+00 1.62250000e+02 3.60490000e+02 ... 4.14770000e+02
        3.77010000e+02 3.10000000e+02]
       [4.19512195e-01 2.05910000e+02 2.68450000e+02 ... 3.51640000e+02
        2.95000000e+02 3.10000000e+02]
       [8.39024390e-01 3.16830000e+02 1.26260000e+02 ... 2.74350000e+02
        3.95300000e+02 3.10000000e+02]
       ...
       [1.98255892e+04 2.64320000e+02 2.43080000e+02 ... 2.25970000e+02
        4.01200000e+02 3.10000000e+02]
       [1.98263928e+04 3.19780000e+02 3.38660000e+02 ... 2.75530000e+02
        5.19200000e+02 3.10000000e+02]
       [1.98271964e+04 2.41310000e+02 4.15360000e+02 ... 2.91460000e+02
        4.63150000e+02 3.10000000e+02]]
    lines.shape = (32526, 7)
    columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    '''

    if verbose:
        start = time.time()

    # If we have a wildcard (*) in the filepath, find all files
    if "*" in filepath:
        filepaths = natsorted(glob(filepath))
        if verbose:
            print(f"Concatenating files:\n  {filepaths}")

    # Otherwise make sure the single file exists
    else:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(textwrap.fill((
                f"The input file path {filepath} does not exist!"
            )))

        filepaths = [filepath]

    lines = convert_adac_forte(filepaths[0])

    # If there are multiple files, concatenate them (and add up the timestamps)
    for i in range(1, len(filepaths)):
        new_lines = convert_adac_forte(filepaths[i])
        new_lines[:, 0] += lines[-1, 0]

        lines = np.vstack((lines, new_lines))

    # Flip Y axis
    lines[:, [2, 5]] = 600 - lines[:, [2, 5]]

    if verbose:
        end = time.time()
        print(f"\nInitialised PEPT data in {end - start:3.3f} s.\n")

    return LineData(lines, sample_size = sample_size, overlap = overlap)
