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


class LineData:
    '''A class for PEPT LoR data iteration, manipulation and visualisation.

    Generally, PEPT LoRs are lines in 3D space, each defined by two points,
    irrespective of the geometry of the scanner used. This class is used
    for LoRs (or any lines!) encapsulation. It can yield samples of the
    `line_data` of an adaptive `sample_size` and `overlap`, without requiring
    additional storage.

    Parameters
    ----------
    line_data : (N, 7) numpy.ndarray
        An (N, 7) numpy array that stores the PEPT LoRs (or any generic set of
        lines) as time and cartesian (3D) coordinates of two points defining each
        line, **in mm**. A row is then [time, x1, y1, z1, x2, y2, z2].
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
        is required to be smaller than `sample_size`. (Default is 0)
    number_of_lines : int
        An `int` that corresponds to len(`line_data`), or the number of
        LoRs stored by `line_data`.
    number_of_samples : int
        An `int` that corresponds to the number of samples that can be
        accessed from the class. It takes `overlap` into consideration.

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size` unless `sample_size` is 0. Overlap
        has to be smaller than `sample_size`. Note that it can also be negative.
    ValueError
        If `line_data` does not have (N, 7) shape.

    Notes
    -----
    The class saves `line_data` as a **contiguous** numpy array for
    efficient access in C functions. It should not be changed after
    instantiating the class.

    '''

    def __init__(
        self,
        line_data,
        sample_size = 200,
        overlap = 0,
        verbose = False
    ):

        if verbose:
            start = time.time()

        # If sample_size != 0 (in which case the class returns all data in one
        # sample), check the `overlap` is not larger or equal to `sample_size`.
        if sample_size < 0:
            raise ValueError('\n[ERROR]: sample_size = {} must be positive (>= 0)'.format(sample_size))
        if sample_size != 0 and overlap >= sample_size:
            raise ValueError('\n[ERROR]: overlap = {} must be smaller than sample_size = {}\n'.format(overlap, sample_size))

        # Initialise the inner parameters of the class
        self._index = 0
        self._sample_size = sample_size
        self._overlap = overlap

        # If `line_data` is not C-contiguous, create a C-contiguous copy
        self._line_data = np.asarray(line_data, order = 'C', dtype = float)

        # Check that line_data has shape (N, 7)
        if self._line_data.ndim != 2 or self._line_data.shape[1] != 7:
            raise ValueError('\n[ERROR]: line_data should have dimensions (N, 7). Received {}\n'.format(self._line_data.shape))

        self._number_of_lines = len(self._line_data)

        if verbose:
            end = time.time()
            print("Initialising the PEPT data took {} seconds\n".format(end - start))


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
    def sample_size(self):
        '''Get the number of lines in one sample returned by the class.

        Returns
        -------
        int
            The sample size (number of lines) in one sample returned by
            the class.

        '''

        return self._sample_size


    @sample_size.setter
    def sample_size(self, new_sample_size):
        '''Change `sample_size` without instantiating a new object

        It also resets the inner index of the class.

        Parameters
        ----------
        new_sample_size : int
            The new sample size. It has to be larger than `overlap`,
            unless it is 0 (in which case all `line_data` will be returned
            as one sample).

        Raises
        ------
        ValueError
            If `overlap` >= `new_sample_size`. Overlap has to be
            smaller than `sample_size`, unless `sample_size` is 0.
            Note that it can also be negative.

        '''

        if new_sample_size < 0:
            raise ValueError('\n[ERROR]: sample_size = {} must be positive (>= 0)'.format(new_sample_size))
        if new_sample_size != 0 and self._overlap >= new_sample_size:
            raise ValueError('\n[ERROR]: overlap = {} must be smaller than new_sample_size = {}\n'.format(self._overlap, new_sample_size))

        self._index = 0
        self._sample_size = new_sample_size


    @property
    def overlap(self):
        '''Get the overlap between every two samples returned by the class.

        Returns
        -------
        int
            The overlap (number of lines) between every two samples  returned by
            the class.

        '''

        return self._overlap


    @overlap.setter
    def overlap(self, new_overlap):
        '''Change `overlap` without instantiating a new object

        It also resets the inner index of the class.

        Parameters
        ----------
        new_overlap : int
            The new overlap. It has to be smaller than `sample_size`, unless
            `sample_size` is 0 (in which case all `line_data` will be returned
            as one sample and so overlap does not play any role).

        Raises
        ------
        ValueError
            If `new_overlap` >= `sample_size`. `new_overlap` has to be
            smaller than `sample_size`, unless `sample_size` is 0.
            Note that it can also be negative.

        '''

        if self._sample_size != 0 and new_overlap >= self._sample_size:
            raise ValueError('\n[ERROR]: new_overlap = {} must be smaller than sample_size = {}\n'.format(new_overlap, self._sample_size))

        self._index = 0
        self._overlap = new_overlap


    @property
    def number_of_samples(self):
        '''Get number of samples, considering overlap.

        If `sample_size == 0`, all data is returned as a single sample,
        and so `number_of_samples` will be 1. Otherwise, it checks the
        number of samples every time it is called, taking `overlap` into
        consideration.

        Returns
        -------
        int
            The number of samples, taking `overlap` into consideration.

        '''
        # If self.sample_size == 0, all data is returned as a single sample
        if self._sample_size == 0:
            return 1

        # If self.sample_size != 0, check there is at least one sample
        if self._number_of_lines >= self._sample_size:
            return (self._number_of_lines - self._sample_size) // (self.sample_size - self.overlap) + 1
        else:
            return 0


    @property
    def number_of_lines(self):
        '''Get the number of lines stored in the class.

        Returns
        -------
        int
            The number of lines stored in `line_data`.

        '''
        return self._number_of_lines


    def sample_n(self, n):
        '''Get sample number n (indexed from 1, i.e. `n > 0`)

        Returns the lines from `line_data` included in sample number
        `n`. Samples are numbered starting from 1.

        Parameters
        ----------
        n : int
            The number of the sample required. Note that `1 <= n <=
            number_of_samples`.

        Returns
        -------
        (, 7) numpy.ndarray
            A shallow copy of the lines from `line_data` included in
            sample number n.

        Raises
        ------
        IndexError
            If `sample_size == 0`, all data is returned as one single
            sample. Raised if `n` is not 1.
        IndexError
            If `n > number_of_samples` or `n <= 0`.

        '''
        if self._sample_size == 0:
            if n == 1:
                return self._line_data
            else:
                raise IndexError("\n\n[ERROR]: Trying to access a non-existent sample (samples are indexed from 1): asked for sample number {}, when there is only 1 sample (sample_size == 0)\n".format(n))
        elif (n > self.number_of_samples) or n <= 0:
            raise IndexError("\n\n[ERROR]: Trying to access a non-existent sample (samples are indexed from 1): asked for sample number {}, when there are {} samples\n".format(n, self.number_of_samples))

        start_index = (n - 1) * (self._sample_size - self._overlap)
        return self._line_data[start_index:(start_index + self._sample_size)]


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
                such that numbers in the format '123,456.78' are well-understood.
            newline : str, optional
                The sequence of characters at the end of every line. The default
                is a new line '\n'

        '''
        np.savetxt(filepath, self._line_data, delimiter = delimiter, newline = newline)


    def plot_all_lines(self, ax = None, color='r', alpha=1.0 ):
        '''Plot all lines using matplotlib

        Given a **mpl_toolkits.mplot3d.Axes3D** axis `ax`, plots all lines on it.

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
        p2 = self._line_data[:, 4:7]

        for i in range(0, self._number_of_lines):
            ax.plot([ p1[i][0], p2[i][0] ],
                    [ p1[i][1], p2[i][1] ],
                    [ p1[i][2], p2[i][2] ],
                    c = color, alpha = alpha)

        return fig, ax


    def plot_all_lines_alt_axes(self, ax, color='r', alpha=1.0):
        '''Plot all lines using matplotlib on PEPT-style axes

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
        p2 = self._line_data[:, 4:7]

        for i in range(0, self._number_of_lines):
            ax.plot([ p1[i][2], p2[i][2] ],
                    [ p1[i][0], p2[i][0] ],
                    [ p1[i][1], p2[i][1] ],
                    c = color, alpha=alpha)

        return fig, ax


    def plot_lines_sample_n(self, n, ax = None, color = 'r', alpha = 1.0):
        '''Plot lines from sample `n` using matplotlib

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
        '''Plot lines from sample `n` using matplotlib on PEPT-style axes

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
        sample_indices = 0,
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
        can be a single sample index (e.g. 0) or an iterable of indices (e.g.
        [1,5,6]).
        Can then be passed to the `plotly.graph_objects.figure.add_trace`
        function or a `PlotlyGrapher` instance using the `add_trace` method.

        Parameters
        ----------
        sample_indices : int or iterable
            The index or indices of the samples of LoRs.
        width : float
            The width of the lines. The default is 2.
        color : str or list-like
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a colorbar list.
            Is ignored if `colorbar` is set to True. For more information, check the Plotly
            documentation. The default is None.
        opacity : float
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque. The default is 0.6.
        colorbar : bool
            If set to True, will color-code the data in the sample column `colorbar_col`.
            Overrides `color` if set to True. The default is True, so that every line has
            a different color.
        colorbar_col : int
            The column in the data samples that will be used to color the points. Only has
            an effect if `colorbar` is set to True. The default is 0 (the first column - time).
        colorbar_title : str
            If set, the colorbar will have this title above. The default is None.

        Returns
        -------
        plotly.graph_objs.Scatter3d
            A Plotly trace of the LoRs.

        '''

        # Check if sample_indices is an iterable collection (list-like)
        # otherwise just "iterate" over the single number
        if not hasattr(sample_indices, "__iter__"):
            sample_indices = [sample_indices]

        marker = dict(
            width = width,
            color = color,
        )

        if colorbar:
            marker['color'] = []
            marker.update(colorscale = "Magma")

            if colorbar_title is not None:
                marker.update(colorbar = dict(title = colorbar_title))

        coords_x = []
        coords_y = []
        coords_z = []

        # For each selected sample include all the lines' coordinates
        for n in sample_indices:
            sample = self[n]

            for line in sample:
                coords_x.extend([line[1], line[4], None])
                coords_y.extend([line[2], line[5], None])
                coords_z.extend([line[3], line[6], None])

                if colorbar:
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


    def __len__(self):
        # Defined so that len(class_instance) returns the number of samples.

        return self.number_of_samples


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

        docstr = "Class instance that inherits from `pept.LineData`.\n\n" + self.__str__() + "\n\n"
        docstr += "Particular cases:\n"
        docstr += " > If sample_size == 0, all line_data is returned as one single sample.\n"
        docstr += " > If overlap >= sample_size, an error is raised.\n"
        docstr += " > If overlap < 0, lines are skipped between samples.\n"

        return docstr


    def __getitem__(self, key):
        # Defined so that samples can be accessed as class_instance[0]

        if self.number_of_samples == 0:
            raise IndexError("Tried to access sample {} (indexed from 0), when there are {} samples".format(key, self.number_of_samples))

        if key >= self.number_of_samples:
            raise IndexError("Tried to access sample {} (indexed from 0), when there are {} samples".format(key, self.number_of_samples))


        while key < 0:
            key += self.number_of_samples

        return self.sample_n(key + 1)


    def __iter__(self):
        # Defined so the class can be iterated as `for sample in class_instance: ...`
        return self


    def __next__(self):
        # sample_size = 0 => return all data
        if self._sample_size == 0:
            self._sample_size = -1
            return self._line_data
        # Use -1 as a flag
        if self._sample_size == -1:
            self._sample_size = 0
            raise StopIteration

        # sample_size > 0 => return slices
        if self._index != 0:
            self._index = self._index + self._sample_size - self.overlap
        else:
            self._index = self._index + self.sample_size


        if self._index > self.number_of_lines:
            self._index = 0
            raise StopIteration

        return self._line_data[(self._index - self._sample_size):self._index]








