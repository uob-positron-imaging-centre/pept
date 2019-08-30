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


# File   : plotly_grapher.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 23.08.2019


'''The *plotly_grapher* module implements Plotly-based visualisation tools
to aid PEPT data analysis and to create publication-ready figures.

The *PlotlyGrapher* class can create and automatically configure 3D subplots
for PEPT data visualisation. The PEPT 3D axes convention has the *y*-axis
pointing upwards, such that the vertical screens of a PEPT scanner represent
the *xy*-plane. The class provides functionality for plotting 3D scatter or
line plots, with optional colorbars.

If you use the `pept` package, you should cite
the following paper: [TODO: paper signature].

'''


import  numpy                                   as          np

import  plotly.graph_objects                    as          go
from    plotly.subplots                         import      make_subplots

import  pept




class PlotlyGrapher:
    '''A class for PEPT data visualisation using Plotly-based 3D plots.

    The *PlotlyGrapher* class can create and automatically configure 3D subplots
    for PEPT data visualisation. It can create and handle any number of subplots,
    automatically configuring them to the ''alternative PEPT 3D axes convention''.
    The PEPT 3D axes convention has the *y*-axis
    pointing upwards, such that the vertical screens of a PEPT scanner represent
    the *xy*-plane. The class provides functionality for plotting 3D scatter or
    line plots, with optional colorbars.

    The class provides easy access to the most common configuration parameters for
    the plots, such as axes limits, subplot titles, colorbar titles, etc. It can
    work with pre-computed Plotly traces (such as the ones from the `pept` base
    classes), as well as with numpy arrays.

    Parameters
    ----------
    rows : int, optional
        The number of rows of subplots. The default is 1.
    cols : int, optional
        The number of columns of subplots. The default is 1.
    xlim : list or numpy.ndarray, optional
        A list of length 2, formatted as `[x_min, x_max]`, where `x_min` is
        the lower limit of the x-axis of all the subplots and `x_max` is the
        upper limit of the x-axis of all the subplots. The default is
        [0, 500], as for the Birmingham PEPT.
    ylim : list or numpy.ndarray, optional
        A list of length 2, formatted as `[y_min, y_max]`, where `y_min` is
        the lower limit of the y-axis of all the subplots and `y_max` is the
        upper limit of the y-axis of all the subplots. The default is
        [0, 500], as for the Birmingham PEPT.
    zlim : list or numpy.ndarray, optional
        A list of length 2, formatted as `[z_min, z_max]`, where `z_min` is
        the lower limit of the z-axis of all the subplots and `z_max` is the
        upper limit of the z-axis of all the subplots. The default is
        [0, 712], a usual screen separation for the Birmingham PEPT.
    subplot_titles : list of str, optional
        A list of the titles of the subplots. The default is a list of empty
        strings.

    Attributes
    ----------
    xlim : list or numpy.ndarray
        A list of length 2, formatted as `[x_min, x_max]`, where `x_min` is
        the lower limit of the x-axis of all the subplots and `x_max` is the
        upper limit of the x-axis of all the subplots.
    ylim : list or numpy.ndarray
        A list of length 2, formatted as `[y_min, y_max]`, where `y_min` is
        the lower limit of the y-axis of all the subplots and `y_max` is the
        upper limit of the y-axis of all the subplots.
    zlim : list or numpy.ndarray
        A list of length 2, formatted as `[z_min, z_max]`, where `z_min` is
        the lower limit of the z-axis of all the subplots and `z_max` is the
        upper limit of the z-axis of all the subplots.
    fig : Plotly Figure instance
        A Plotly Figure instance, with any number of subplots (as defined by
        `rows` and `cols`) pre-configured for PEPT data. It is created when
        calling `create_figure`.

    Raises
    ------
    ValueError
        If `rows` < 1 or `cols` < 1
    TypeError
        If `xlim`, `ylim` or `zlim` are not lists of length 2.

    Notes
    -----
    The class constructor (calling PlotlyGrapher()) is separate from the figure
    creation, so that one can have instances of both the `PlotlyGrapher` and the
    Plotly figure it creates. An example call would be:

    >>> grapher = PlotlyGrapher()
    >>> fig = grapher.create_figure()

    '''

    def __init__(self,
                 rows = 1, cols = 1,
                 xlim = [0, 500], ylim = [0, 500], zlim = [0, 712],
                 subplot_titles = ['  ']):

        if rows < 1 or cols < 1:
            raise ValueError("\n[ERROR]: The number of rows and cols have to be larger than 1\n")

        self._rows = rows
        self._cols = cols

        if len(xlim) != 2 or len(ylim) != 2 or len(zlim) != 2:
            raise TypeError("\n[ERROR]: xlim, ylim and zlim need to be lists of length 2, formatted as xlim = [x_min, x_max] etc.\n")

        self._xlim = xlim
        self._ylim = ylim
        self._zlim = zlim

        self._subplot_titles = subplot_titles
        self._subplot_titles.extend(['  '] * (rows * cols - len(subplot_titles)))

        self._fig = None


    def create_figure(self):
        '''Create a Plotly figure, pre-configured for PEPT data.

        This function creates a Plotly figure with any number of subplots,
        as given in the class instantiation call. It pre-configures them to have
        the *y*-axis pointing upwards, as per the PEPT 3D axes convention. It
        also sets the axes limits and labels.

        Returns
        -------
        fig : Plotly Figure instance
            A Plotly Figure instance, with any number of subplots (as defined when instantiating
            the class) pre-configured for PEPT data.

        '''

        specs = [[{"type": "scatter3d"}] * self._cols] * self._rows

        self._fig = make_subplots(rows = self._rows, cols = self._cols,
                    specs = specs, subplot_titles = self._subplot_titles,
                    horizontal_spacing = 0.005, vertical_spacing = 0.05)

        self._fig['layout'].update(margin = dict(l=0,r=0,b=30,t=30), showlegend = False)

        # For every subplot (scene), set axes' ratios and limits
        # Also set the y axis to point upwards
        # Plotly naming convention of scenes: 'scene', 'scene2', etc.
        for i in range(self._rows):
            for j in range(self._cols):
                if i == j == 0:
                    scene = 'scene'
                else:
                    scene = 'scene{}'.format(i * self._cols + j + 1)

                # Justify subplot title on the left
                self._fig.layout.annotations[i * self._cols + j].update(x = (j + 0.15) / self._cols)
                self._fig['layout'][scene].update(aspectmode = 'manual',
                                                 aspectratio = {'x': 1, 'y': 1, 'z': 1},
                                                 camera = {'up': {'x': 0, 'y': 1, 'z':0},
                                                           'eye': {'x': 1, 'y': 1, 'z': 1}},
                                                 xaxis = {'range': self._xlim,
                                                          'title': {'text': "<i>x</i> (mm)"}},
                                                 yaxis = {'range': self._ylim,
                                                          'title': {'text': "<i>y</i> (mm)"}},
                                                 zaxis = {'range': self._zlim,
                                                          'title': {'text': "<i>z</i> (mm)"}}
                                                 )

        return self._fig


    @property
    def xlim(self):
        '''The upper and lower limits of the *x*-axis.

        The property is defined as a list of length 2, formatted as
        [x_min, x_max]

        Returns
        -------
        xlim : list or numpy.ndarray
            A list of length 2 representing the upper and lower limits.

        '''

        return self._xlim


    @xlim.setter
    def xlim(self, new_xlim):
        '''Set the lower and upper boundaries of the *x*-axis

        Parameters
        ----------
        new_xlim : list or numpy.ndarray
            A list of length 2 representing the upper and lower limits of
            the *x*-axis, formatted as [x_min, x_max]

        Raises
        ------
        TypeError
            If `xlim`, `ylim` or `zlim` are not lists of length 2.
        AttributeError
            If no figure was created (i.e. create_figure was not called)

        '''

        if len(new_xlim) != 2:
            raise TypeError("\n[ERROR]: xlim, ylim and zlim need to be lists of length 2, formatted as xlim = [x_min, x_max] etc.\n")

        if self._fig is None:
            raise AttributeError("\n[ERROR]: No figure was created! First run create_figure()\n")

        self._xlim = new_xlim

        # For every subplot (scene), update axes' limits
        # Plotly naming convention of scenes: 'scene', 'scene2', etc.
        for i in range(self._rows):
            for j in range(self._cols):
                if i == j == 0:
                    scene = 'scene'
                else:
                    scene = 'scene{}'.format(i * self._cols + j + 1)

                self._fig['layout'][scene].update(xaxis = {'range': self._xlim})


    @property
    def ylim(self):
        '''The upper and lower limits of the *y*-axis.

        The property is defined as a list of length 2, formatted as
        [y_min, y_max]

        Returns
        -------
        ylim : list or numpy.ndarray
            A list of length 2 representing the upper and lower limits.

        '''

        return self._ylim


    @ylim.setter
    def ylim(self, new_ylim):
        '''Set the lower and upper boundaries of the *y*-axis

        Parameters
        ----------
        new_ylim : list or numpy.ndarray
            A list of length 2 representing the upper and lower limits of
            the *y*-axis, formatted as [y_min, y_max]

        Raises
        ------
        TypeError
            If `xlim`, `ylim` or `zlim` are not lists of length 2.
        AttributeError
            If no figure was created (i.e. create_figure was not called)

        '''

        if len(new_ylim) != 2:
            raise TypeError("\n[ERROR]: xlim, ylim and zlim need to be lists of length 2, formatted as xlim = [x_min, x_max] etc.\n")

        if self._fig is None:
            raise AttributeError("\n[ERROR]: No figure was created! First run create_figure()\n")

        self._ylim = new_ylim

        # For every subplot (scene), update axes' limits
        # Plotly naming convention of scenes: 'scene', 'scene2', etc.
        for i in range(self._rows):
            for j in range(self._cols):
                if i == j == 0:
                    scene = 'scene'
                else:
                    scene = 'scene{}'.format(i * self._cols + j + 1)

                self._fig['layout'][scene].update(yaxis = {'range': self._ylim})


    @property
    def zlim(self):
        '''The upper and lower limits of the *z*-axis.

        The property is defined as a list of length 2, formatted as
        [z_min, z_max]

        Returns
        -------
        zlim : list or numpy.ndarray
            A list of length 2 representing the upper and lower limits.

        '''

        return self._zlim


    @zlim.setter
    def zlim(self, new_zlim):
        '''Set the lower and upper boundaries of the *z*-axis

        Parameters
        ----------
        new_zlim : list or numpy.ndarray
            A list of length 2 representing the upper and lower limits of
            the *z*-axis, formatted as [z_min, z_max]

        Raises
        ------
        TypeError
            If `xlim`, `ylim` or `zlim` are not lists of length 2.
        AttributeError
            If no figure was created (i.e. create_figure was not called)

        '''

        if len(new_zlim) != 2:
            raise TypeError("\n[ERROR]: xlim, ylim and zlim need to be lists of length 2, formatted as xlim = [x_min, x_max] etc.\n")

        if self._fig is None:
            raise AttributeError("\n[ERROR]: No figure was created! First run create_figure()\n")

        self._zlim = new_zlim

        # For every subplot (scene), update axes' limits
        # Plotly naming convention of scenes: 'scene', 'scene2', etc.
        for i in range(self._rows):
            for j in range(self._cols):
                if i == j == 0:
                    scene = 'scene'
                else:
                    scene = 'scene{}'.format(i * self._cols + j + 1)

                self._fig['layout'][scene].update(zaxis = {'range': self._zlim})


    @property
    def fig(self):
        '''Return the Plotly Figure instance

        This is useful for manual configuration of the Plotly figure
        stored in this class.

        Returns
        -------
        fig : Plotly Figure instance
            The Plotly Figure instance stored in the class, containing all
            data and options set.

        '''

        return self._fig


    def add_data_as_trace(self, data, row = 1, col = 1, size = 2, color = None):
        '''Create a Plotly trace of `data` and add it to the figure.

        Creates Scatter3d Plotly traces from the rows of `data` and adds it to
        the subplot determined by `row` and `col`.

        Parameters
        ----------
        data : (N, M >= 4) numpy.ndarray
            The expected data row: [time, x, y, z, etc.]
        row : int, optional
            The row of the subplot to add a trace to. The default is 1.
        col : int, optional
            The column of the subplot to add a trace to. The default is 1.
        size : int, optional
            The size of the trace markers. The default is 2.
        color : str, optional
            The color of the trace markers. The default is `None`.

        Raises
        ------
        TypeError
            If `data` is not a numpy.ndarray with shape (N, M), where M >= 4

        '''

        if data.ndim != 2 or data.shape[1] < 4:
            raise TypeError("\n[ERROR]: data should be a numpy.ndarray with shape (N, M), where M >= 4\n")

        trace = go.Scatter3d(
            x = data[:, 1],
            y = data[:, 2],
            z = data[:, 3],
            mode = 'markers',
            marker = dict(
                size = size,
                color = color,
                opacity = 0.8
            )
        )

        self._fig.add_trace(trace, row = row, col = col)


    def add_data_as_trace_colorbar(self, data,
                                   row = 1, col = 1,
                                   title_colorbar = None,
                                   size = 3, colorbar_col = -1):
        '''Create a colour-coded Plotly trace of `data` and add it to the figure.

        Creates Scatter3d Plotly traces from the rows of `data` and adds it to
        the subplot determined by `row` and `col`. It colour-codes the data in
        terms of a given column `colorbar_col` from the data.

        Parameters
        ----------
        data : (N, M >= 4) numpy.ndarray
            The expected data row: [time, x, y, z, etc.]
        row : int, optional
            The row of the subplot to add a trace to. The default is 1.
        col : int, optional
            The column of the subplot to add a trace to. The default is 1.
        size : int, optional
            The size of the trace markers. The default is 2.
        title_colorbar : str, optional
            The title to display on the colorbar on the right of the plot.
            The default is `None`.
        colorbar_col : int, optional
            The column in `data` based on which the markers will be colour-coded.
            The default is -1, the last column.

        Raises
        ------
        TypeError
            If `data` is not a numpy.ndarray with shape (N, M), where M >= 4
        ValueError
            If `colorbar_col` is larger than the number of columns in `data`

        '''

        if data.ndim != 2 or data.shape[1] < 4:
            raise TypeError("\n[ERROR]: data should be a numpy.ndarray with shape (N, M), where M >= 4\n")

        if colorbar_col > data.shape[1]:
            raise ValueError("\n[ERROR]: colorbar_col was larger than the number of columns in data\n")

        if title_colorbar != None:
            colorbar = dict(title= title_colorbar)
        else:
            colorbar = dict()

        trace = go.Scatter3d(
            x = data[:, 1],
            y = data[:, 2],
            z = data[:, 3],
            mode = 'markers',
            marker = dict(
                size = size,
                color = data[:, colorbar_col],   # set color to sample size
                colorscale = 'Magma',     # choose a colorscale
                colorbar = colorbar,
                opacity = 0.8
            )
        )

        self._fig.add_trace(trace, row = row, col = col)


    def add_data_as_trace_line(self, data, row = 1, col = 1, width = 4, color = None):
        '''Create a Plotly trace of `data` and add it to the figure.

        Creates Scatter3d Plotly traces as continuous lines from the rows of
        `data` and adds it to the subplot determined by `row` and `col`.

        Parameters
        ----------
        data : (N, M >= 4) numpy.ndarray
            The expected data row: [time, x, y, z, etc.]
        row : int, optional
            The row of the subplot to add a trace to. The default is 1.
        col : int, optional
            The column of the subplot to add a trace to. The default is 1.
        width : int, optional
            The width of the trace lines. The default is 4.
        color : str, optional
            The color of the trace markers. The default is `None`.

        Raises
        ------
        TypeError
            If `data` is not a numpy.ndarray with shape (N, M), where M >= 4

        '''

        if data.ndim != 2 or data.shape[1] < 4:
            raise TypeError("\n[ERROR]: data should be a numpy.ndarray with shape (N, M), where M >= 4\n")

        trace = go.Scatter3d(
            x = data[:, 1],
            y = data[:, 2],
            z = data[:, 3],
            mode = 'lines',
            line = dict(
                width = width,
                color = color
            )
        )

        self._fig.add_trace(trace, row = row, col = col)


    def add_data_as_trace_lines(self, data, row = 1, col = 1, width = 2, color = None):
        '''Create Plotly traces for individual lines in `data` and add them to the figure.

        Creates Scatter3d Plotly traces as continuous lines from the rows of
        `data` and adds it to the subplot determined by `row` and `col`. It expects
        LoR-like data, where every line is defined by two points.

        Parameters
        ----------
        data : (N, M >= 7) numpy.ndarray
            The expected data row: [time, x1, y1, z1, x2, y2, z2, etc.]
        row : int, optional
            The row of the subplot to add a trace to. The default is 1.
        col : int, optional
            The column of the subplot to add a trace to. The default is 1.
        width : int, optional
            The width of the trace lines. The default is 4.
        color : str, optional
            The color of the trace markers. The default is `None`.

        Raises
        ------
        TypeError
            If `data` is not a numpy.ndarray with shape (N, M), where M >= 7

        '''

        if data.ndim != 2 or data.shape[1] < 7:
            raise TypeError("\n[ERROR]: data should be a numpy.ndarray with shape (N, M), where M >= 7\n")


        # data is a list of lines, each defined by two points
        # data row [time, x1, y1, z1, x2, y2, z2]
        for line in data:
            self._fig.add_trace(
                go.Scatter3d(
                    x = [ line[1], line[4] ],
                    y = [ line[2], line[5] ],
                    z = [ line[3], line[6] ],
                    mode = 'lines',
                    opacity = 0.6,
                    line = dict(
                        width = width,
                        color = color
                    )
                ),
                row = row,
                col = col
			)


    def add_trace(self, trace, row = 1, col = 1):
        '''Add a precomputed Plotly trace to a given subplot.

        The equivalent of the Plotly figure.add_trace method.

        Parameters
        ----------
        trace : Plotly trace (Scatter3d)
            A precomputed Plotly trace
        row : int, optional
            The row of the subplot to add a trace to. The default is 1.
        col : int, optional
            The column of the subplot to add a trace to. The default is 1.

        '''

        # Add precomputed trace
        self._fig.add_trace(trace, row = row, col = col)


    def add_traces(self, traces, row = 1, col = 1):
        '''Add a list of precomputed Plotly traces to a given subplot.

        The equivalent of the Plotly figure.add_traces method.

        Parameters
        ----------
        traces : list [ Plotly trace (Scatter3d) ]
            A list of precomputed Plotly traces
        row : int, optional
            The row of the subplot to add a trace to. The default is 1.
        col : int, optional
            The column of the subplot to add a trace to. The default is 1.

        '''

        # Add precomputed traces
        self._fig.add_traces(traces, rows=[row]*len(traces), cols=[col]*len(traces))


    def show(self):
        '''Show the Plotly figure.

        '''

        self._fig.show()




