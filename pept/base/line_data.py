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


# File   : line_data.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 19.08.2019


import  time
import  numpy                   as      np

import  plotly.graph_objects    as      go

import  matplotlib
import  matplotlib.pyplot       as      plt
from    matplotlib.colors       import  Normalize
from    mpl_toolkits.mplot3d    import  Axes3D

from    .iterable_samples       import  IterableSamples


class LineData(IterableSamples):
    '''A class for PEPT LoR data iteration, manipulation and visualisation.

    Generally, PEPT LoRs are lines in 3D space, each defined by two points,
    irrespective of the geometry of the scanner used. This class is used
    for LoRs (or any lines!) encapsulation. It can yield samples of the
    `lines` of an adaptive `sample_size` and `overlap`, without requiring
    additional storage.

    Parameters
    ----------
    lines : (N, 7) numpy.ndarray
        An (N, 7) numpy array that stores the PEPT LoRs (or any generic set of
        lines) as time and cartesian (3D) coordinates of two points defining
        each line, **in mm**. A row is then [time, x1, y1, z1, x2, y2, z2].
    sample_size : int, optional
        An `int`` that defines the number of lines that should be returned when
        iterating over `lines`. A `sample_size` of 0 yields all the data as
        one single sample. Default is 0.
    overlap : int, optional
        An `int` that defines the overlap between two consecutive samples that
        are returned when iterating over `lines`. An overlap of 0 means
        consecutive samples, while an overlap of (`sample_size` - 1) means
        incrementing the samples by one. A negative overlap means skipping
        values between samples. An error is raised if `overlap` is larger than
        or equal to `sample_size`. Default is 0.
    verbose : bool, optional
        An option that enables printing the time taken for the initialisation
        of an instance of the class. Useful when reading large files (10gb
        files for PEPT data is not unheard of). Default is False.

    Attributes
    ----------
    lines : (N, 7) numpy.ndarray
        An (N, 7) numpy array that stores the PEPT LoRs as time and cartesian
        (3D) coordinates of two points defining a line, **in mm**. Each row is
        then `[time, x1, y1, z1, x2, y2, z2]`.
    sample_size : int
        An `int` that defines the number of lines that should be returned when
        iterating over `lines`. Default is 0.
    overlap : int
        An `int` that defines the overlap between two consecutive samples that
        are returned when iterating over `lines`. An overlap of 0 implies
        consecutive samples, while an overlap of (`sample_size` - 1) implies
        incrementing the samples by one. A negative overlap means skipping
        values between samples. It is required to be smaller than
        `sample_size`. Default is 0.
    number_of_lines : int
        An `int` that corresponds to len(`lines`), or the number of LoRs
        stored by `lines`.
    number_of_samples : int
        An `int` that corresponds to the number of samples that can be accessed
        from the class. It takes `overlap` into consideration.

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size` unless `sample_size` is 0. Overlap
        has to be smaller than `sample_size`. Note that it can also be
        negative.
    ValueError
        If `lines` has fewer than 7 columns.

    Notes
    -----
    This class is made for LoRs that do not have Time of Flight data, such that
    every row in `lines` is comprised of a single timestamp and the points'
    coordinates: [time, x1, y1, z1, x2, y2, z2]. If your PET / PEPT scanner
    has Time of Flight data, use the `LineDataToF` class.

    The class saves `lines` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the number of rows or columns after instantiating the class.

    '''

    def __init__(
        self,
        lines,
        sample_size = 0,
        overlap = 0,
        verbose = False
    ):

        if verbose:
            start = time.time()

        # If `lines` is not C-contiguous, create a C-contiguous copy
        self._lines = np.asarray(lines, order = 'C', dtype = float)

        # Check that lines has at least 7 columns.
        if self._lines.ndim != 2 or self._lines.shape[1] < 7:
            raise ValueError((
                "\n[ERROR]: `lines` should have dimensions (M, N), where "
                f"N >= 7. Received {self._lines.shape}.\n"
            ))

        self._number_of_lines = len(self._lines)

        # Call the IterableSamples constructor to make the class iterable in
        # samples with overlap.
        IterableSamples.__init__(self, sample_size, overlap)

        if verbose:
            end = time.time()
            print(f"Initialising the line data took {end - start} seconds.\n")


    @property
    def lines(self):
        '''Get the lines stored in the class.

        Returns
        -------
        (, 7) numpy.ndarray
            A memory view of the lines stored in `lines`.

        '''
        return self._lines


    @property
    def data_samples(self):
        '''Implement property for the IterableSamples parent class. See its
        documentation for more information.

        '''

        return self._lines


    @property
    def data_length(self):
        '''Implement property for the IterableSamples parent class. See its
        documentation for more information.

        '''
        return self._number_of_lines


    @property
    def number_of_lines(self):
        '''Get the number of lines stored in the class.

        Returns
        -------
        int
            The number of lines stored in `lines`.

        '''
        return self._number_of_lines


    def to_csv(self, filepath, delimiter = '  ', newline = '\n'):
        '''Write `lines` to a CSV file

        Write all LoRs stored in the class to a CSV file.

        Parameters
        ----------
            filepath : filename or file handle
                If filepath is a path (rather than file handle), it is relative
                to where python is called.
            delimiter : str, optional
                The delimiter between values. The default is two spaces '  ',
                such that numbers in the format '123,456.78' are
                well-understood.
            newline : str, optional
                The sequence of characters at the end of every line. The
                default is a new line '\n'

        '''
        np.savetxt(filepath, self._lines, delimiter = delimiter,
                   newline = newline)


    def plot_all_lines(self, ax = None, color='r', alpha=1.0 ):
        '''Plot all lines using Matplotlib.

        Given a **mpl_toolkits.mplot3d.Axes3D** axis `ax`, plots all lines on
        it.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.

        color : matplotlib color option (default 'r')

        alpha : matplotlib opacity option (default 1.0)

        Returns
        -------

        fig, ax : matplotlib figure and axes objects

        Note
        ----
        Plotting all lines in the case of large LoR arrays is *very*
        computationally intensive. For large arrays (> 10000), plotting
        individual samples using `plot_lines_sample_n` is recommended.

        '''
        if ax == None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()

        p1 = self._lines[:, 1:4]
        p2 = self._lines[:, 4:7]

        for i in range(0, self._number_of_lines):
            ax.plot([ p1[i][0], p2[i][0] ],
                    [ p1[i][1], p2[i][1] ],
                    [ p1[i][2], p2[i][2] ],
                    c = color, alpha = alpha)

        return fig, ax


    def plot_all_lines_alt_axes(self, ax, color='r', alpha=1.0):
        '''Plot all lines using matplotlib on PEPT-style axes.

        Given a **mpl_toolkits.mplot3d.Axes3D** axis `ax`, plots all lines on
        the PEPT-style convention: **x** is *parallel and horizontal* to the
        screens, **y** is *parallel and vertical* to the screens, **z** is
        *perpendicular* to the screens. The mapping relative to the
        Cartesian coordinates would then be: (x, y, z) -> (z, x, y)

        Parameters
        ----------
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.

        color : matplotlib color option (default 'r')

        alpha : matplotlib opacity option (default 1.0)

        Returns
        -------

        fig, ax : matplotlib figure and axes objects

        Note
        ----
        Plotting all lines in the case of large LoR arrays is *very*
        computationally intensive. For large arrays (> 10000), plotting
        individual samples using `plot_lines_sample_n_alt_axes` is recommended.

        '''
        if ax == None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()


        p1 = self._lines[:, 1:4]
        p2 = self._lines[:, 4:7]

        for i in range(0, self._number_of_lines):
            ax.plot([ p1[i][2], p2[i][2] ],
                    [ p1[i][0], p2[i][0] ],
                    [ p1[i][1], p2[i][1] ],
                    c = color, alpha=alpha)

        return fig, ax


    def plot_lines_sample_n(self, n, ax = None, color = 'r', alpha = 1.0):
        '''Plot lines from sample `n` using Matplotlib.

        Given a **mpl_toolkits.mplot3d.Axes3D** axis `ax`, plots all lines
        from sample number `n`.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.

        sampleN : int
            The number of the sample to be plotted.

        color : matplotlib color option (default 'r')

        alpha : matplotlib opacity option (default 1.0)

        Returns
        -------

        fig, ax : matplotlib figure and axes objects

        '''
        if ax == None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()

        sample = self.sample_n(n)
        for i in range(0, len(sample)):
            ax.plot([ sample[i][1], sample[i][4] ],
                    [ sample[i][2], sample[i][5] ],
                    [ sample[i][3], sample[i][6] ],
                    c = color, alpha = alpha)

        return fig, ax


    def plot_lines_sample_n_alt_axes(self, n, ax=None, color='r', alpha=1.0):
        '''Plot lines from sample `n` using matplotlib on PEPT-style axes.

        Given a **mpl_toolkits.mplot3d.Axes3D** axis `ax`, plots all lines from
        sample number sampleN on the PEPT-style coordinates convention:
        **x** is *parallel and horizontal* to the screens, **y** is
        *parallel and vertical* to the screens, **z** is *perpendicular*
        to the screens. The mapping relative to the Cartesian coordinates
        would then be: (x, y, z) -> (z, x, y)

        Parameters
        ----------
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.
        n : int
            The number of the sample to be plotted.

        color : matplotlib color option (default 'r')

        alpha : matplotlib opacity option (default 1.0)

        Returns
        -------

        fig, ax : matplotlib figure and axes objects

        '''
        if ax == None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()

        sample = self.sample_n(n)
        for i in range(0, len(sample)):
            ax.plot([ sample[i][3], sample[i][6] ],
                    [ sample[i][1], sample[i][4] ],
                    [ sample[i][2], sample[i][5] ],
                    c = color, alpha = alpha)

        return fig, ax


    def lines_trace(
        self,
        sample_indices = ...,
        width = 2,
        color = None,
        opacity = 0.6,
        colorbar = True,
        colorbar_col = 0,
        colorbar_title = None
    ):
        '''Get a Plotly trace for all the lines in selected samples.

        Creates a `plotly.graph_objects.Scatter3d` object for all the lines
        included in the samples selected by `sample_indices`. `sample_indices`
        may be a single sample index (e.g. 0), an iterable of indices (e.g.
        [1,5,6]) or an Ellipsis (`...`) for all samples.

        Can then be passed to the `plotly.graph_objects.figure.add_trace`
        function or a `PlotlyGrapher` instance using the `add_trace` method.

        Parameters
        ----------
        sample_indices : int or iterable or Ellipsis
            The index or indices of the samples of LoRs. An `int` signifies the
            sample index, an iterable (list-like) signifies multiple sample
            indices, while an Ellipsis (`...`) signifies all samples. The
            default is `...` (all lines).
        width : float
            The width of the lines. The default is 2.
        color : str or list-like
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Overrides `colorbar` if set. For more information,
            check the Plotly documentation. The default is None.
        opacity : float
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque. The default is 0.6.
        colorbar : bool
            If set to True, will color-code the data in the sample column
            `colorbar_col`. Is overridden if `color` is set. The default is
            True, so that every line has a different color.
        colorbar_col : int
            The column in the data samples that will be used to color the
            points. Only has an effect if `colorbar` is set to True. The
            default is 0 (the first column - time).
        colorbar_title : str
            If set, the colorbar will have this title above. The default is
            None.

        Returns
        -------
        plotly.graph_objs.Scatter3d
            A Plotly trace of the LoRs.

        '''
        # Check if sample_indices is an iterable collection (list-like)
        # otherwise just "iterate" over the single number or Ellipsis.
        if not hasattr(sample_indices, "__iter__"):
            sample_indices = [sample_indices]

        marker = dict(
            width = width,
            color = color,
        )

        if colorbar:
            if color is None:
                marker['color'] = []
            marker.update(colorscale = "Magma")

            if colorbar_title is not None:
                marker.update(colorbar = dict(title = colorbar_title))

        coords_x = []
        coords_y = []
        coords_z = []

        # For each selected sample include all the lines' coordinates
        for n in sample_indices:
            # If an Ellipsis was received, then include all lines.
            if n is Ellipsis:
                sample = self.lines
            else:
                sample = self[n]

            for line in sample:
                coords_x.extend([line[1], line[4], None])
                coords_y.extend([line[2], line[5], None])
                coords_z.extend([line[3], line[6], None])

                if colorbar and color is None:
                    marker['color'].extend(3 * [line[colorbar_col]])

        trace = go.Scatter3d(
            x = coords_x,
            y = coords_y,
            z = coords_z,
            mode = 'lines',
            opacity = opacity,
            line = marker
        )

        return trace


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"number_of_lines =   {self.number_of_lines}\n\n"
            f"sample_size =       {self._sample_size}\n"
            f"overlap =           {self._overlap}\n"
            f"number_of_samples = {self.number_of_samples}\n\n"
            f"lines = \n{self._lines}"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPR
        docstr = (
            "Class instance that inherits from `pept.LineData`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n----------\n"
            f"{self.__str__()}\n\n"
            "Particular Cases\n----------------\n"
            " > If sample_size == 0, all `lines` are returned as a "
               "single sample.\n"
            " > If overlap >= sample_size, an error is raised.\n"
            " > If overlap < 0, lines are skipped between samples.\n"
        )

        return docstr


