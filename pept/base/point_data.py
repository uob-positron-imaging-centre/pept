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


# File   : point_data.py
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


class PointData(IterableSamples):
    '''A class for generic PEPT data iteration, manipulation and visualisation.

    This class is used to encapsulate points. It can yield samples of the
    `point_data` of an adaptive `sample_size` and `overlap`, without requiring
    additional storage.

    Parameters
    ----------
    point_data : (N, M) numpy.ndarray
        An (N, M >= 4) numpy array that stores points (or any generic 2D set of
        data). It expects that the first column is time, followed by cartesian
        (3D) coordinates of points **in mm**, followed by any extra information
        the user needs. A row is then [time, x, y, z, etc].
    sample_size : int, optional
        An `int`` that defines the number of points that should be returned
        when iterating over `point_data`. A `sample_size` of 0 yields all the
        data as one single sample. Default is 0.
    overlap : int, optional
        An `int` that defines the overlap between two consecutive samples that
        are returned when iterating over `point_data`. An overlap of 0 means
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
    point_data : (N, M) numpy.ndarray
        An (N, M >= 4) numpy array that stores the points as time, followed by
        cartesian (3D) coordinates of the point **in mm**, followed by any
        extra information. Each row is then `[time, x, y, z, etc]`.
    sample_size : int
        An `int` that defines the number of lines that should be returned when
        iterating over `point_data`. Default is 0.
    overlap : int
        An `int` that defines the overlap between two consecutive samples that
        are returned when iterating over `point_data`. An overlap of 0 means
        consecutive samples, while an overlap of (`sample_size` - 1) means
        incrementing the samples by one. A negative overlap means skipping
        values between samples. It is required to be smaller than
        `sample_size`. Default is 0.
    number_of_points : int
        An `int` that corresponds to len(`point_data`), or the number of points
        stored by `point_data`.
    number_of_samples : int
        An `int` that corresponds to the number of samples that can be accessed
        from the class, taking the `overlap` into consideration.

    Raises
    ------
    ValueError
        If `overlap` >= `sample_size`. Overlap is required to be smaller than
        `sample_size`, unless `sample_size` is 0. Note that it can also be
        negative.
    ValueError
        If `line_data` does not have (N, M) shape, where M >= 4.

    Notes
    -----
    The class saves `point_data` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the number of rows or columns after instantiating the class.

    '''


    def __init__(
        self,
        point_data,
        sample_size = 0,
        overlap = 0,
        verbose = False
    ):

        if verbose:
            start = time.time()

        # If `point_data` is not C-contiguous, create a C-contiguous copy.
        self._point_data = np.asarray(point_data, order = 'C', dtype = float)

        # Check that point_data has at least 4 columns.
        if self._point_data.ndim != 2 or self._point_data.shape[1] < 4:
            raise ValueError((
                "\n[ERROR]: `point_data` should have dimensions (M, N), where "
                f"N >= 4. Received {self._point_data.shape}.\n"
            ))

        self._number_of_points = len(self._point_data)

        # Call the IterableSamples constructor to make the class iterable in
        # samples with overlap.
        IterableSamples.__init__(self, sample_size, overlap)

        if verbose:
            end = time.time()
            print(f"Initialising the PEPT data took {end - start} seconds.\n")


    @property
    def point_data(self):
        '''Get the points stored in the class.

        Returns
        -------
        (M, N) numpy.ndarray
            A memory view of the points stored in `point_data`.

        '''
        return self._point_data


    @property
    def data_samples(self):
        '''Implemented property for the IterableSamples parent class. See its
        documentation for more information.

        '''
        return self._point_data


    @property
    def data_length(self):
        '''Implemented property for the IterableSamples parent class. See its
        documentation for more information.

        '''
        return self._number_of_points


    @property
    def number_of_points(self):
        '''Get the number of points stored in the class.

        Returns
        -------
        int
            The number of points stored in `point_data`.

        '''
        return self._number_of_points


    def to_csv(self, filepath, delimiter = '  ', newline = '\n'):
        '''Write `point_data` to a CSV file

        Write all points (and any extra data) stored in the class to a CSV file.

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
        np.savetxt(filepath, self._point_data, delimiter = delimiter, newline = newline)


    def plot_all_points(self, ax = None):
        '''Plot all points using matplotlib

        Given a **mpl_toolkits.mplot3d.Axes3D** axis, plots all points on it.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.

        Returns
        -------
        fig, ax : matplotlib figure and axes objects

        Note
        ----
        Plotting all points in the case of large LoR arrays is *very*
        computationally intensive. For large arrays (> 10000), plotting
        individual samples using `plot_points_sample_n` is recommended.

        '''
        if ax == None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()

        # Scatter x, y, z, [color]

        x = self._point_data[:, 1],
        y = self._point_data[:, 2],
        z = self._point_data[:, 3],

        color = self._point_data[:, -1],

        cmap = plt.cm.magma
        color_array = cmap(colour_data)

        ax.scatter(x,y,z,c=color_array[0])

        return fig, ax


    def plot_all_points_alt_axes(self, ax = None ):
        '''Plot all points using matplotlib on PEPT-style axes

        Given a **mpl_toolkits.mplot3d.Axes3D** axis, plots all points on
        the PEPT-style convention: **x** is *parallel and horizontal* to the
        screens, **y** is *parallel and vertical* to the screens, **z** is
        *perpendicular* to the screens. The mapping relative to the
        Cartesian coordinates would then be: (x, y, z) -> (z, x, y)

        Parameters
        ----------
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.

        Returns
        -------
        fig, ax : matplotlib figure and axes objects

        Note
        ----
        Plotting all points in the case of large LoR arrays is *very*
        computationally intensive. For large arrays (> 10000), plotting
        individual samples using `plot_lines_sample_n_alt_axes` is recommended.

        '''
        if ax == None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()

        # Scatter x, y, z, [color]

        x = self._point_data[:, 1]
        y = self._point_data[:, 2]
        z = self._point_data[:, 3]

        color = self._point_data[:, -1]

        cmap = plt.cm.magma
        color_array = cmap(color)

        ax.scatter(z,x,y,c=color_array[0])

        return fig, ax


    def plot_points_sample_n(self, n, ax=None):
        '''Plot points from sample `n` using matplotlib

        Given a **mpl_toolkits.mplot3d.Axes3D** axis, plots all points
        from sample number `n`.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3D.Axes3D object
            The 3D matplotlib-based axis for plotting.
        n : int
            The number of the sample to be plotted.

        Returns
        -------

        fig, ax : matplotlib figure and axes objects

        '''
        if ax == None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()

        # Scatter x, y, z, [color]

        sample = self.sample_n(n)

        x = sample[:, 1]
        y = sample[:, 2]
        z = sample[:, 3]

        color = sample[:, -1]

        cmap = plt.cm.magma
        color_array = cmap(color)

        ax.scatter(z,x,y,c=color_array[0])

        return fig, ax


    def plot_points_sample_n_alt_axes(self, n, ax=None):
        '''Plot points from sample `n` using matplotlib on PEPT-style axes

        Given a **mpl_toolkits.mplot3d.Axes3D** axis, plots all points from
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

        Returns
        -------

        fig, ax : matplotlib figure and axes objects

        '''
        if ax == None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.gcf()

        # Scatter x, y, z, [color]

        sample = self.sample_n(n)

        x = sample[:, 1]
        y = sample[:, 2]
        z = sample[:, 3]

        color = sample[:, -1]

        cmap = plt.cm.magma
        color_array = cmap(color)

        ax.scatter(z,x,y,c=color_array[0])

        return fig, ax


    def points_trace(
        self,
        sample_indices = 0,
        size = 2,
        color = None,
        opacity = 0.8,
        colorbar = False,
        colorbar_col = -1,
        colorbar_title = None
    ):
        '''Get a Plotly trace for all points in selected samples, with possible
        color-coding.

        Returns a `plotly.graph_objects.Scatter3d` trace containing all points
        included in in the samples selected by `sample_indices`.
        `sample_indices` can be a single sample index (e.g. 0) or an iterable
        of indices (e.g. [1,5,6]).

        Can then be passed to the `plotly.graph_objects.figure.add_trace`
        function or a `PlotlyGrapher` instance using the `add_trace` method.

        Parameters
        ----------
        sample_indices : int or iterable
            The index or indices of the samples of LoRs. The default is 0
            (the first sample).
        size : float
            The marker size of the points. The default is 2.
        color : str or list-like
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Is ignored if `colorbar` is set to True. For more
            information, check the Plotly documentation. The default is None.
        opacity : float
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque. The default is 0.8.
        colorbar : bool
            If set to True, will color-code the data in the sample column
            `colorbar_col`. Overrides `color` if set to True. The default is
            False.
        colorbar_col : int
            The column in the data samples that will be used to color the
            points. Only has an effect if `colorbar` is set to True. The
            default is -1 (the last column).
        colorbar_title : str
            If set, the colorbar will have this title above. The default is
            None.

        Returns
        -------
        plotly.graph_objs.Scatter3d
            A Plotly trace of the points.

        '''
        # Check if sample_indices is an iterable collection (list-like)
        # otherwise just "iterate" over the single number
        if not hasattr(sample_indices, "__iter__"):
            sample_indices = [sample_indices]

        coords_x = []
        coords_y = []
        coords_z = []

        marker = dict(
            size = size,
            color = color,
            opacity = opacity
        )

        if colorbar:
            marker['color'] = []
            marker.update(colorscale = "Magma")

            if colorbar_title is not None:
                marker.update(colorbar = dict(title = colorbar_title))

        # For each selected sample include all the needed coordinates
        for n in sample_indices:
            sample = self[n]

            coords_x.extend(sample[:, 1])
            coords_y.extend(sample[:, 2])
            coords_z.extend(sample[:, 3])

            if colorbar == True:
                marker['color'].extend(sample[:, colorbar_col])

        trace = go.Scatter3d(
            x = coords_x,
            y = coords_y,
            z = coords_z,
            mode = "markers",
            marker = marker
        )

        return trace


    def __str__(self):
        # Shown when calling print(class)
        docstr = ""

        docstr += "number_of_points =  {}\n\n".format(self.number_of_points)
        docstr += "sample_size =       {}\n".format(self._sample_size)
        docstr += "overlap =           {}\n".format(self._overlap)
        docstr += "number_of_samples = {}\n\n".format(self.number_of_samples)
        docstr += "point_data = \n"
        docstr += self._point_data.__str__()

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPR

        docstr = "Class instance that inherits from `pept.PointData`.\n\n" + self.__str__() + "\n\n"
        docstr += "Particular cases:\n"
        docstr += " > If sample_size == 0, all point_data is returned as one single sample.\n"
        docstr += " > If overlap >= sample_size, an error is raised.\n"
        docstr += " > If overlap < 0, points are skipped between samples.\n"

        return docstr


