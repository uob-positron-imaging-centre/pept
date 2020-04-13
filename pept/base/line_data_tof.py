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


# File   : line_data_tof.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 09.04.2020


import  time
import  numpy                   as      np

import  plotly.graph_objects    as      go

import  matplotlib
import  matplotlib.pyplot       as      plt
from    matplotlib.colors       import  Normalize
from    mpl_toolkits.mplot3d    import  Axes3D

from    .iterable_samples       import  IterableSamples
import  pept


class LineDataToF(IterableSamples):
    '''A class for PEPT-ToF LoR data iteration, manipulation and visualisation.

    Generally, PEPT LoRs are lines in 3D space, each defined by two points,
    irrespective of the geometry of the scanner used. This class is used
    for LoRs (or any lines!) encapsulation. It can yield samples of the
    `line_data` of an adaptive `sample_size` and `overlap`, without requiring
    additional storage.

    Note that the spatial and temporal data must have self-consistent units
    (i.e. millimetres and milliseconds OR metres and seconds). Also, besides
    the expected [time1, x1, y1, z1, time2, x2, y2, z2] columns, any extra
    information can be appended after the two points.

    Parameters
    ----------
    line_data : (N, M >= 8) numpy.ndarray
        An (N, M >= 8) numpy array that stores the PEPT LoRs with Time of
        Flight (ToF) data (or any generic set of lines with two timestamps) as
        individual time and cartesian (3D) coordinates of two points defining
        each line, **in mm**. A row is then [t1, x1, y1, z1, t2, x2, y2, z2].
    sample_size : int, optional
        An `int`` that defines the number of lines that should be returned when
        iterating over `line_data`. A `sample_size` of 0 yields all the data as
        one single sample. Default is 0.
    overlap : int, optional
        An `int` that defines the overlap between two consecutive samples that
        are returned when iterating over `line_data`. An overlap of 0 means
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
    line_data : (N, M >= 8) numpy.ndarray
        An (N, M >= 8) numpy array that stores the PEPT LoRs as individual
        times (**in ms**) and cartesian (3D) coordinates of two points defining
        a line, **in mm**. Each row is then
        `[time1, x1, y1, z1, time2, x2, y2, z2]`.
    sample_size : int
        An `int` that defines the number of lines that should be returned when
        iterating over `line_data`. Default is 0.
    overlap : int
        An `int` that defines the overlap between two consecutive samples that
        are returned when iterating over `line_data`. An overlap of 0 implies
        consecutive samples, while an overlap of (`sample_size` - 1) implies
        incrementing the samples by one. A negative overlap means skipping
        values between samples. It is required to be smaller than
        `sample_size`. Default is 0.
    number_of_lines : int
        An `int` that corresponds to len(`line_data`), or the number of LoRs
        stored by `line_data`.
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
        If `line_data` has fewer than 8 columns.

    Notes
    -----
    This class is made for LoRs that also have Time of Flight data, such that
    every row in `line_data` is comprised of two points defining a line along
    with their individual timestamps: [time1, x1, y1, z1, time2, x2, y2, z2].
    If your PET / PEPT scanner does not have Time of Flight data, use the
    `LineData` class.

    The class saves `line_data` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the number of rows or columns after instantiating the class.

    '''

    def __init__(
        self,
        line_data,
        sample_size = 0,
        overlap = 0,
        append_tofpoints = True,
        verbose = False
    ):

        if verbose:
            start = time.time()

        # If `line_data` is not C-contiguous, create a C-contiguous copy.
        self._line_data = np.asarray(line_data, order = 'C', dtype = float)

        # Check that line_data has at least 8 columns.
        if self._line_data.ndim != 2 or self._line_data.shape[1] < 8:
            raise ValueError((
                "\n[ERROR]: `line_data` should have dimensions (M, N), where "
                f"N >= 8. Received {self._line_data.shape}.\n"
            ))

        self._number_of_lines = len(self._line_data)

        # Call the IterableSamples constructor to make the class iterable in
        # samples with overlap.
        IterableSamples.__init__(self, sample_size, overlap)

        # If set, calculate and append the ToF data to the LoR data
        if append_tofpoints:
            self.append_tofpoints()

        if verbose:
            end = time.time()
            print(f"Initialising the line data took {end - start} seconds.\n")


    @property
    def line_data(self):
        '''Get the lines stored in the class.

        Returns
        -------
        (, 7) numpy.ndarray
            A memory view of the lines stored in `line_data`.

        '''
        return self._line_data


    @property
    def data_samples(self):
        '''Implement property for the IterableSamples parent class. See its
        documentation for more information.

        '''

        return self._line_data


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
            The number of lines stored in `line_data`.

        '''
        return self._number_of_lines


    def get_tofpoints(self, as_array = False):
        '''Calculate and return the tofpoints calculated from the ToF data.

        The tofpoints include both the time and locations calculated from the
        Time of Flight (ToF) data. They can be returned either as a simple
        numpy array (`as_array = True`) or wrapped in a pept.PointData
        (default) for ease of plotting iteration.

        Parameters
        ----------
        as_array : bool, optional
            If set to `True`, the calculated tofpoints will be returned as a
            numpy.ndarray. Otherwise, return an instance of `pept.PointData`.

        Returns
        -------
        pept.PointData or numpy.ndarray
            If `as_array` is set to `True`, the tofpoints are returned as a
            numpy.ndarray. Otherwise (the default option), they are wrapped in
            a `pept.PointData` instance.

        '''
        # The two points defining the LoR
        t1 = self._line_data[:, 0]
        p1 = self._line_data[:, 1:4]

        t2 = self._line_data[:, 4]
        p2 = self._line_data[:, 5:8]

        # Speed of light (mm / ms)
        c = 299792458

        # The ratio (P1 - tofpoint) / (P1 - P2) for all rows
        distance_ratio = 0.5 - 0.5 / np.linalg.norm(p2 - p1, axis = 1) * \
                         c * (t2 - t1)

        # [:, np.newaxis] = transform row vector to column vector (i.e. 2D
        # array with one column)
        tof_locations = p1 + (p2 - p1) * distance_ratio[:, np.newaxis]
        tof_time = t1 - np.linalg.norm(tof_locations - p1, axis = 1) / c

        tofpoints = np.hstack((tof_time[:, np.newaxis], tof_locations))

        if not as_array:
            tofpoints = pept.PointData(tofpoints)

        return tofpoints


    def append_tofpoints(self):
        '''Calculate and append the tofpoints to the LoR data.

        The tofpoints are appended to the LoR data stored in this class.
        Therefore, if the initial `self.line_data` has a row [t1, x1, y1, z1,
        t2, x2, y2, z2], then after calling this function the `self.line_data`
        row will be [t1, x1, y1, z1, t2, x2, y2, z2, t_tof, x_tof, y_tof,
        z_tof].

        Note that if any extra columns are included in `self.line_data`, they
        will not be affected; the tofpoints will simply be appended after them.

        '''

        tofpoints = self.get_tofpoints(as_array = True)

        # Append the ToF data to the LoR data
        self._line_data = np.hstack((self._line_data, tofpoints))


    def to_csv(self, filepath, delimiter = '  ', newline = '\n'):
        '''Write `line_data` to a CSV file

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
        np.savetxt(filepath, self._line_data, delimiter = delimiter, newline = newline)


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

        p1 = self._line_data[:, 1:4]
        p2 = self._line_data[:, 5:8]

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


        p1 = self._line_data[:, 1:4]
        p2 = self._line_data[:, 5:8]

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
            ax.plot([ sample[i][1], sample[i][5] ],
                    [ sample[i][2], sample[i][6] ],
                    [ sample[i][3], sample[i][7] ],
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
            ax.plot([ sample[i][3], sample[i][7] ],
                    [ sample[i][1], sample[i][5] ],
                    [ sample[i][2], sample[i][6] ],
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
                sample = self.line_data
            else:
                sample = self[n]

            for line in sample:
                coords_x.extend([line[1], line[5], None])
                coords_y.extend([line[2], line[6], None])
                coords_z.extend([line[3], line[7], None])

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
        docstr = ""

        docstr += "number_of_lines =   {}\n\n".format(self.number_of_lines)
        docstr += "sample_size =       {}\n".format(self._sample_size)
        docstr += "overlap =           {}\n".format(self._overlap)
        docstr += "number_of_samples = {}\n\n".format(self.number_of_samples)
        docstr += "line_data = \n"
        docstr += self._line_data.__str__()

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPR

        docstr = "Class instance that inherits from `pept.LineDataToF`.\n\n" + \
            self.__str__() + "\n\n"
        docstr += "Particular cases:\n"
        docstr += (" > If sample_size == 0, all line_data is returned as one"
                   "single sample.\n")
        docstr += " > If overlap >= sample_size, an error is raised.\n"
        docstr += " > If overlap < 0, lines are skipped between samples.\n"

        return docstr


