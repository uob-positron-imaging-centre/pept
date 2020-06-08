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

# File   : parallel_screens.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.08.2019


import  time
import  numpy           as      np

from    pept            import  LineData
from    pept.utilities  import  read_csv




class ParallelScreens(LineData):
    '''A subclass of `LineData` that initialises PEPT data for parallel screens
    detectors from an input data file or array.

    Provides the same functionality as the `LineData` class while initialising
    `line_data` from  **PEPT scanners with two parallel screens**. That is,
    each LoR is defined by two 2D points on two screens separated by a given
    distance.

    **The expected data columns in the file is `[time, x1, y1, x2, y2]`**. This
    is automatically transformed into the standard `line_data` format with
    columns being `[time, x1, y1, z1, x2, y2, z2]`, where `z1 = 0` and
    `z2 = separation`.

    `ParallelScreens` can be initialised with a predefined numpy array of LoRs
    or read data from a `.csv` or `.a0n` file or equivalent.

    Parameters
    ----------
    filepath_or_array : [str, pathlib.Path, IO] or numpy.ndarray-like (N, 5)
        A path to a file to be read or an array for initialisation. A path is a
        string with the (absolute or relative) path to the data file from which
        the PEPT data will be read. It should include the full file name, along
        with the extension (.csv, .a01, etc.).
    screen_separation : float
        The separation (in *mm*) between the two PEPT screens corresponding to
        the `z` coordinate of the second point defining each line. The
        attribute `line_data`, with columns
        `[time, x1, y1, z1, x2, y2, z2]`, will have `z1 = 0` and
        `z2 = separation`.
    sample_size : int, default 200
        An `int`` that defines the number of lines that should be returned when
        iterating over `line_data`. A `sample_size` of 0 yields all the data as
        one single sample.
    overlap : int, default 0
        An `int` that defines the overlap between two consecutive samples that
        are returned when iterating over `line_data`. An overlap of 0 implies
        consecutive samples, while an overlap of (`sample_size` - 1) means
        incrementing the samples by one. A negative overlap means skipping
        values between samples. An error is raised if `overlap` is larger than
        or equal to `sample_size`.
    skiprows : int, default 0
        The number of rows to skip from the beginning of the data file. Useful
        when the data file includes a header of text that should be skipped.
    max_rows : int, optional
        The maximum number of rows that will be read from the data file.
    verbose : bool, default True
        An option that enables printing the time taken for the initialisation
        of an instance of the class. Useful when reading large files (10gb
        files for PEPT data is not unheard of).

    Attributes
    ----------
    line_data : (N, 7) numpy.ndarray
        An (N, 7) numpy array that stores the PEPT LoRs as time and
        cartesian (3D) coordinates of two points defining a line, **in mm**.
        Each row is then `[time, x1, y1, z1, x2, y2, z2]`.
    sample_size : int
        An `int` that defines the number of lines that should be
        returned when iterating over `line_data`.
    overlap : int
        An `int` that defines the overlap between two consecutive
        samples that are returned when iterating over `line_data`.
        An overlap of 0 means consecutive samples, while an overlap
        of (`sample_size` - 1) means incrementing the samples by one.
        A negative overlap means skipping values between samples. It
        has to be smaller than `sample_size`.
    number_of_lines : int
        An `int` that corresponds to len(`line_data`), or the number of
        LoRs stored by `line_data`.

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size`. Overlap has to be smaller than
        `sample_size`. Note that it can also be negative.
    ValueError
        If the data file does not have the (N, M >= 5) shape.

    Notes
    -----
    The class saves `line_data` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the number of rows or columns after instantiating the class.

    '''

    def __init__(
        self,
        filepath_or_array,
        screen_separation,
        sample_size = 200,
        overlap = 0,
        skiprows = None,
        nrows = None,
        verbose = True
    ):

        if verbose:
            start = time.time()

        # Check wheter input is a valid `pandas.read_csv` filepath or a numpy
        # array in a "Better To Ask Forgiveness Than Permission" way.
        try:
            # Try to read the LoR data from `filepath_or_array`.
            # Check if an error is raised when reading file lines using
            line_data = read_csv(
                filepath_or_array,
                skiprows = skiprows,
                nrows = nrows
            )
        except ValueError:
            # Seems like it is an array!
            line_data = np.asarray(
                filepath_or_array,
                order = "C",
                dtype = float
            )

        # line_data cols: [time, X1, Y1, X2, Y2]
        # Verify that line_data has shape (N, M >= 5)
        if line_data.ndim != 2 or line_data.shape[1] < 5:
            raise ValueError((
                "\n[ERROR]: line_data should have dimensions (N, M) where "
                f"M >= 5. Received {line_data.shape}.\n"
            ))

        # Add Z1 and Z2 columns => [time, X1, Y1, Z1, X2, Y2, Z2]
        # Z1 = 0
        line_data = np.insert(line_data, 3, 0.0, axis = 1)

        # Z2 = `separation`
        line_data = np.insert(line_data, 6, screen_separation, axis = 1)

        # Call the constructor of the superclass `LineData` to initialise all
        # the inner parameters of the class (_index, etc.)
        LineData.__init__(
            self,
            line_data,
            sample_size = sample_size,
            overlap = overlap,
            verbose = False
        )

        if verbose:
            end = time.time()
            print(f"Initialising the PEPT data took {end - start} seconds.\n")




