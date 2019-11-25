#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
#
#    Copyright (C) 2019 Andrei Leonard Nicusan
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
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.08.2019


import  time
import  numpy   as      np
from    pept    import  LineData


class ParallelScreens(LineData):
    '''A subclass of `LineData` that reads PEPT data for parallel screens detectors.

    Provides the same functionality as the `LineData` class while
    initialising `line_data` from a given file. This is a helper class
    for **PEPT scanners with two parallel screens**.

    `ParallelScreens` can read data from a `.csv` or `.a0n` file or equivalent.
    **The expected data row in the file is `[time, x1, y1, x2, y2]`**. This will
    then be automatically transformed into the standard `line_data` format
    with every row being `[time, x1, y1, z1, x2, y2, z2]`, where `z1 = 0`
    and `z2 = separation`.

    Parameters
    ----------
    data_file : str
        A string with the (absolute or relative) path to the data file
        from which the PEPT data will be read. It should include the
        full file name, along with the extension (.csv, .a01, etc.)
    sample_size : int, optional
        An `int`` that defines the number of lines that should be
        returned when iterating over `line_data`. A `sample_size` of 0
        yields all the data as one single sample. (Default is 200)
    overlap : int, optional
        An `int` that defines the overlap between two consecutive
        samples that are returned when iterating over `line_data`.
        An overlap of 0 means consecutive samples, while an overlap
        of (`sample_size` - 1) means incrementing the samples by one.
        A negative overlap means skipping values between samples. An
        error is raised if `overlap` is larger than or equal to
        `sample_size`. (Default is 0)
    separation : float, optional
        The separation (in *mm*) between the two PEPT screens corresponding
        to the `z` coordinate of the second point defining each line.
        The attribute `line_data`, with each row being
        `[time, x1, y1, z1, x2, y2, z2]`, will have `z1 = 0` and
        `z2 = separation`. (Default is 712)
    skiprows : int, optional
        The number of rows to skip from the beginning of the data file.
        Useful when the data file includes a header of text that should
        be skipped. (Default is 0)
    max_rows : int, optional
        The maximum number of rows that will be read from the data file.
        (Default is `None`)
    verbose : bool, optional
        An option that enables printing the time taken for the
        initialisation of an instance of the class. Useful when
        reading large files (10gb files for PEPT data is not unheard
        of). (Default is True)

    Attributes
    ----------
    line_data : (N, 7) numpy.ndarray
        An (N, 7) numpy array that stores the PEPT LoRs as time and
        cartesian (3D) coordinates of two points defining a line, **in mm**.
        Each row is then `[time, x1, y1, z1, x2, y2, z2]`.
    sample_size : int
        An `int` that defines the number of lines that should be
        returned when iterating over `line_data`. (Default is 200)
    overlap : int
        An `int` that defines the overlap between two consecutive
        samples that are returned when iterating over `line_data`.
        An overlap of 0 means consecutive samples, while an overlap
        of (`sample_size` - 1) means incrementing the samples by one.
        A negative overlap means skipping values between samples. It
        has to be smaller than `sample_size`. (Default is 0)
    number_of_lines : int
        An `int` that corresponds to len(`line_data`), or the number of
        LoRs stored by `line_data`.

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size`. Overlap has to be smaller than
        `sample_size`. Note that it can also be negative.
    ValueError
        If the data file does not have (N, 5) shape.

    Notes
    -----
    The class saves `line_data` as a **contiguous** numpy array for
    efficient access in C functions. It should not be changed after
    instantiating the class.

    '''

    def __init__(self,
                 data_file,
                 separation,
                 sample_size = 200,
                 overlap = 0,
                 skiprows = 0,
                 max_rows = None,
                 verbose = True):

        if verbose:
            start = time.time()

        # Read the LoR data from `data_file`
        # Row: [time, X1, Y1, X2, Y2]
        line_data = np.loadtxt(data_file, skiprows = skiprows, max_rows = max_rows)

        # Verify that line_data has shape (N, 5)
        if line_data.ndim != 2 or line_data.shape[1] != 5:
            raise ValueError('\n[ERROR]: line_data (that was read from data_file) should have dimensions (N, 5). Received {}\n'.format(line_data.shape))

        number_of_lines = len(line_data)

        # Add Z1 and Z2 columns => [time, X1, Y1, Z1, X2, Y2, Z2]
        # Z1 = 0
        line_data = np.insert(line_data, 3, np.zeros(number_of_lines), axis = 1)
        # Z2 = `separation`
        line_data = np.append(line_data, separation * np.ones((number_of_lines, 1)), axis = 1)

        # Call the constructor of the superclass `LineData` to initialise all
        # the inner parameters of the class (_index, etc.)
        super().__init__(line_data,
                         sample_size = sample_size,
                         overlap = overlap,
                         verbose = False)

        if verbose:
            end = time.time()
            print("Initialising the PEPT data took {} seconds\n".format(end - start))




