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

If you use the `pept` package, we ask you to cite the following paper:

    Nicuşan AL, Windows-Yule CR. Positron emission particle tracking
    using machine learning. Review of Scientific Instruments.
    2020 Jan 1;91(1):013329.
    https://doi.org/10.1063/1.5129251

'''


import  numpy                   as          np

import  plotly.graph_objects    as          go
from    plotly.subplots         import      make_subplots


class PlotlyGrapher:
    '''A class for PEPT data visualisation using Plotly-based 3D graphs.

    The **PlotlyGrapher** class can create and automatically configure an
    arbitrary number of 3D subplots for PEPT data visualisation. They are by
    default set to use the *alternative PEPT 3D axes convention* - having the
    *y*-axis pointing upwards, such that the vertical screens of a PEPT scanner
    represent the *xy*-plane.

    This class can be used to draw 3D scatter or line plots, with optional
    colour-coding using extra data columns (e.g. relative tracer activity).

    It also provides easy access to the most common configuration parameters
    for the plots, such as axes limits, subplot titles, colorbar titles, etc.
    It can work with pre-computed Plotly traces (such as the ones from the
    `pept` base classes), as well as with numpy arrays.

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
        [0, 500], a usual screen separation for the Birmingham PEPT.
    subplot_titles : list of str, optional
        A list of the titles of the subplots - e.g. ["plot a)", "plot b)"]. The
        default is a list of empty strings.

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
        `rows` and `cols`) pre-configured for PEPT data.

    Raises
    ------
    ValueError
        If `rows` < 1 or `cols` < 1
    TypeError
        If `xlim`, `ylim` or `zlim` are not lists of length 2.

    Example use
    -----------
    The figure is created when instantiating the class

    >>> lors = LineData(raw_lors...)
    >>> grapher = PlotlyGrapher()

    # Using a pre-computed trace from a LineData class:
    >>> grapher.add_trace(lors.lines_trace(colorbar = True))

    # Creating a trace based on a nummpy array:
    >>> sample = lors[0]
    >>> grapher.add_lines(sample, colorbar = True)

    # Showing the plot:
    >>> grapher.show()

    '''

    def __init__(
        self,
        rows = 1,
        cols = 1,
        xlim = None,#[0, 500],
        ylim = None,#[0, 500],
        zlim = None,#[0, 500],
        subplot_titles = ["  "]
    ):

        rows = int(rows)
        cols = int(cols)

        if rows < 1 or cols < 1:
            raise ValueError((
                "\n[ERROR]: The number of rows and cols have to be larger "
                f"than 1. Received rows = {rows}; cols = {cols}.\n"
            ))

        self._rows = rows
        self._cols = cols

        if xlim is not None:
            xlim = np.asarray(xlim, dtype = float)

            if xlim.ndim != 1 or xlim.shape[0] != 2:
                raise TypeError((
                    "\n[ERROR]: xlim needs to be a list of length 2, formatted"
                    f" as xlim = [x_min, x_max]. Received {xlim}.\n"
                ))

        if ylim is not None:
            ylim = np.asarray(ylim, dtype = float)

            if ylim.ndim != 1 or ylim.shape[0] != 2:
                raise TypeError((
                    "\n[ERROR]: ylim needs to be a list of length 2, formatted"
                    f" as ylim = [y_min, y_max]. Received {ylim}.\n"
                ))

        if zlim is not None:
            zlim = np.asarray(zlim, dtype = float)

            if zlim.ndim != 1 or zlim.shape[0] != 2:
                raise TypeError((
                    "\n[ERROR]: zlim needs to be a list of length 2, formatted"
                    f" as zlim = [z_min, z_max]. Received {zlim}.\n"
                ))

        self._xlim = xlim
        self._ylim = ylim
        self._zlim = zlim

        self._subplot_titles = subplot_titles
        # Pad the subplot titles that were not set with empty strings.
        self._subplot_titles.extend(['  '] * (rows * cols -
                                              len(subplot_titles)))

        self._fig = self.create_figure()


    def create_figure(self):
        '''Create a Plotly figure, pre-configured for PEPT data.

        This function creates a Plotly figure with an arbitrary number of
        subplots, as given in the class instantiation call. It configures them
        to have the *y*-axis pointing upwards, as per the PEPT 3D axes
        convention. It also sets the axes limits and labels.

        Returns
        -------
        fig : Plotly Figure instance
            A Plotly Figure instance, with any number of subplots (as defined
            when instantiating the class) pre-configured for PEPT data.

        '''

        specs = [[{"type": "scatter3d"}] * self._cols] * self._rows

        self._fig = make_subplots(
            rows = self._rows,
            cols = self._cols,
            specs = specs,
            subplot_titles = self._subplot_titles,
            horizontal_spacing = 0.005,
            vertical_spacing = 0.05
        )

        self._fig['layout'].update(
            margin = dict(l = 0, r = 0, b = 30, t = 30),
            showlegend = False
        )

        # For every subplot (scene), set axes' ratios and limits
        # Also set the y axis to point upwards
        # Plotly naming convention of scenes: 'scene', 'scene2', etc.
        for i in range(self._rows):
            for j in range(self._cols):
                if i == j == 0:
                    scene = "scene"
                else:
                    scene = "scene{}".format(i * self._cols + j + 1)

                # Justify subplot title on the left
                self._fig.layout.annotations[i * self._cols + j].update(
                    x = (j + 0.15) / self._cols
                )

                self._fig["layout"][scene].update(
                    xaxis = dict(
                        range = self._xlim,
                        title = dict(text = "<i>x</i> (mm)")
                    ),
                    yaxis = dict(
                        range = self._ylim,
                        title = dict(text = "<i>y</i> (mm)")
                    ),
                    zaxis = dict(
                        range = self._zlim,
                        title = dict(text = "<i>z</i> (mm)")
                    ),
                    aspectmode = "manual",
                    aspectratio = dict(x = 1, y = 1, z = 1),
                    camera = dict(
                        up = dict(x = 0, y = 1, z = 0),
                        eye = dict(x = 1, y = 1, z = 1)
                    ),
                )

        return self._fig


    @property
    def xlim(self):
        '''The upper and lower limits of the *x*-axis.

        The property is defined as a numpy array of length 2, formatted as
        [x_min, x_max]

        Returns
        -------
        xlim : numpy.ndarray
            A numpy array of length 2 representing the upper and lower limits.

        '''

        return self._xlim


    @xlim.setter
    def xlim(self, xlim):
        '''Set the lower and upper boundaries of the *x*-axis.

        Parameters
        ----------
        xlim : list or numpy.ndarray
            A list-like of length 2 representing the upper and lower limits of
            the *x*-axis, formatted as [x_min, x_max]

        Raises
        ------
        TypeError
            If `xlim` is not a list-like of length 2.

        '''

        xlim = np.asarray(xlim, dtype = float)

        if xlim.ndim != 1 or xlim.shape[0] != 2:
            raise TypeError((
                "\n[ERROR]: xlim needs to be a list of length 2, formatted as"
                f"xlim = [x_min, x_max]. Received {xlim}.\n"
            ))

        self._xlim = xlim

        # For every subplot (scene), update axes' limits
        # Plotly naming convention of scenes: 'scene', 'scene2', etc.
        for i in range(self._rows):
            for j in range(self._cols):
                if i == j == 0:
                    scene = 'scene'
                else:
                    scene = 'scene{}'.format(i * self._cols + j + 1)

                self._fig['layout'][scene].update(
                    xaxis = dict(range = self._xlim)
                )


    @property
    def ylim(self):
        '''The upper and lower limits of the *y*-axis.

        The property is defined as a numpy array of length 2, formatted as
        [y_min, y_max]

        Returns
        -------
        ylim : numpy.ndarray
            A numpy array of length 2 representing the upper and lower limits.

        '''

        return self._ylim


    @ylim.setter
    def ylim(self, ylim):
        '''Set the lower and upper boundaries of the *y*-axis.

        Parameters
        ----------
        ylim : list or numpy.ndarray
            A list-like of length 2 representing the upper and lower limits of
            the *y*-axis, formatted as [y_min, y_max]

        Raises
        ------
        TypeError
            If `ylim` is not a list-like of length 2.

        '''

        ylim = np.asarray(ylim, dtype = float)

        if ylim.ndim != 1 or ylim.shape[0] != 2:
            raise TypeError((
                "\n[ERROR]: ylim needs to be a list of length 2, formatted as"
                f"ylim = [y_min, y_max]. Received {ylim}.\n"
            ))

        self._ylim = ylim

        # For every subplot (scene), update axes' limits
        # Plotly naming convention of scenes: 'scene', 'scene2', etc.
        for i in range(self._rows):
            for j in range(self._cols):
                if i == j == 0:
                    scene = 'scene'
                else:
                    scene = 'scene{}'.format(i * self._cols + j + 1)

                self._fig['layout'][scene].update(
                    yaxis = dict(range = self._ylim)
                )


    @property
    def zlim(self):
        '''The upper and lower limits of the *z*-axis.

        The property is defined as a numpy array of length 2, formatted as
        [z_min, z_max]

        Returns
        -------
        zlim : numpy.ndarray
            A numpy array of length 2 representing the upper and lower limits.

        '''

        return self._zlim


    @zlim.setter
    def zlim(self, zlim):
        '''Set the lower and upper boundaries of the *z*-axis

        Parameters
        ----------
        zlim : list or numpy.ndarray
            A list-like of length 2 representing the upper and lower limits of
            the *z*-axis, formatted as [z_min, z_max]

        Raises
        ------
        TypeError
            If `zlim` is not a list-like of length 2.

        '''

        zlim = np.asarray(zlim, dtype = float)

        if zlim.ndim != 1 or zlim.shape[0] != 2:
            raise TypeError((
                "\n[ERROR]: zlim needs to be a list of length 2, formatted as"
                f"zlim = [z_min, z_max]. Received {zlim}.\n"
            ))

        self._zlim = zlim

        # For every subplot (scene), update axes' limits
        # Plotly naming convention of scenes: 'scene', 'scene2', etc.
        for i in range(self._rows):
            for j in range(self._cols):
                if i == j == 0:
                    scene = 'scene'
                else:
                    scene = 'scene{}'.format(i * self._cols + j + 1)

                self._fig['layout'][scene].update(
                    zaxis = dict(range = self._zlim)
                )


    @property
    def fig(self):
        '''Return the Plotly Figure instance.

        This is useful for manual configuration of the Plotly figure
        stored in this class.

        Returns
        -------
        fig : Plotly Figure instance
            The Plotly Figure instance stored in the class, containing all
            data and options set.

        '''

        return self._fig


    def add_points(
        self,
        points,
        row = 1,
        col = 1,
        size = 2,
        color = None,
        opacity = 0.8,
        colorbar = False,
        colorbar_col = -1,
        colorbar_title = None
    ):
        '''Create and plot a trace for all points in a numpy array, with
        possible color-coding.

        Creates a `plotly.graph_objects.Scatter3d` object for all the points
        included in the numpy array `points` and adds it to the subplot
        determined by `row` and `col`.

        The expected data row is [time, x1, y1, z1, ...].

        Parameters
        ----------
        points : (M, N >= 4) numpy.ndarray
            The expected data row: [time, x1, y1, z1, etc.]
        row : int, optional
            The row of the subplot to add a trace to. The default is 1.
        col : int, optional
            The column of the subplot to add a trace to. The default is 1.
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
            If set to True, will color-code the data in the `points` column
            `colorbar_col`. Overrides `color` if set to True. The default is
            False.
        colorbar_col : int
            The column in `points` that will be used to color the points. Only
            has an effect if `colorbar` is set to True. The default is -1 (the
            last column).
        colorbar_title : str
            If set, the colorbar will have this title above. The default is
            None.

        Raises
        ------
        TypeError
            If `points` is not a numpy.ndarray with shape (M, N), where N >= 4.

        '''

        points = np.asarray(points, dtype = float)

        # Check that lines has shape (M, 4)
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError((
                "\n[ERROR]: line_data should have dimensions (M, N), where "
                "N >= 4. Received {}\n".format(lines.shape)
            ))

        coords_x = points[:, 1]
        coords_y = points[:, 2]
        coords_z = points[:, 3]

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

        if colorbar == True:
            marker['color'].extend(points[:, colorbar_col])

        trace = go.Scatter3d(
            x = coords_x,
            y = coords_y,
            z = coords_z,
            mode = "markers",
            marker = marker
        )

        self._fig.add_trace(trace, row = row, col = col)




    def add_lines(
        self,
        lines,
        row = 1,
        col = 1,
        width = 2,
        color = None,
        opacity = 0.6,
        colorbar = True,
        colorbar_col = 0,
        colorbar_title = None
    ):
        '''Create and plot a trace for individual lines in a numpy array.

        Creates a `plotly.graph_objects.Scatter3d` object for all the lines
        included in the numpy array `lines` and adds it to the subplot
        determined by `row` and `col`.

        It expects LoR-like data, where each line is defined by two points. The
        expected data row is [time, x1, y1, z1, x2, y2, z2, ...].

        Parameters
        ----------
        lines : (M, N >= 7) numpy.ndarray
            The expected data row: [time, x1, y1, z1, x2, y2, z2, etc.]
        row : int, optional
            The row of the subplot to add a trace to. The default is 1.
        col : int, optional
            The column of the subplot to add a trace to. The default is 1.
        width : float
            The width of the lines. The default is 2.
        color : str or list-like
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Is ignored if `colorbar` is set to True. For more
            information, check the Plotly documentation. The default is None.
        opacity : float
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque. The default is 0.6.
        colorbar : bool
            If set to True, will color-code the data in the `lines` column
            `colorbar_col`. Overrides `color` if set to True. The default is
            True, so that every line has a different color.
        colorbar_col : int
            The column in the data samples that will be used to color the
            points. Only has an effect if `colorbar` is set to True. The
            default is 0 (the first column - time).
        colorbar_title : str
            If set, the colorbar will have this title above. The default is
            None.

        Raises
        ------
        TypeError
            If `lines` is not a numpy.ndarray with shape (M, N), where N >= 7.

        '''

        lines = np.asarray(lines, dtype = float)

        # Check that lines has shape (N, 7)
        if lines.ndim != 2 or lines.shape[1] < 7:
            raise ValueError((
                "\n[ERROR]: line_data should have dimensions (M, N), where "
                "N >= 7. Received {}\n".format(lines.shape)
            ))

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

        for line in lines:
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

        self._fig.add_trace(trace, row = row, col = col)


    def add_lines_tof(
        self,
        lines,
        row = 1,
        col = 1,
        width = 2,
        color = None,
        opacity = 0.6,
        colorbar = True,
        colorbar_col = 0,
        colorbar_title = None
    ):
        '''Create and plot a trace for individual lines with ToF data in a
        numpy array.

        Creates a `plotly.graph_objects.Scatter3d` object for all the lines
        included in the numpy array `lines` and adds it to the subplot
        determined by `row` and `col`.

        It expects LoR-like data with Time of Flight (ToF) data, where each
        line is defined by two points, each having individual timestamps. The
        expected data row is [time1, x1, y1, z1, time2, x2, y2, z2, ...].

        Parameters
        ----------
        lines : (M, N >= 8) numpy.ndarray
            The expected data row: [time1, x1, y1, z1, time2, x2, y2, z2, etc.]
        row : int, optional
            The row of the subplot to add a trace to. The default is 1.
        col : int, optional
            The column of the subplot to add a trace to. The default is 1.
        width : float
            The width of the lines. The default is 2.
        color : str or list-like
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Is ignored if `colorbar` is set to True. For more
            information, check the Plotly documentation. The default is None.
        opacity : float
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque. The default is 0.6.
        colorbar : bool
            If set to True, will color-code the data in the `lines` column
            `colorbar_col`. Overrides `color` if set to True. The default is
            True, so that each line has a different color.
        colorbar_col : int
            The column in the data samples that will be used to color the
            points. Only has an effect if `colorbar` is set to True. The
            default is 0 (the first column - time1).
        colorbar_title : str
            If set, the colorbar will have this title above. The default is
            None.

        Raises
        ------
        TypeError
            If `lines` is not a numpy.ndarray with shape (M, N), where N >= 8.

        '''

        lines = np.asarray(lines, dtype = float)

        # Check that lines has shape (N, 8)
        if lines.ndim != 2 or lines.shape[1] < 8:
            raise ValueError((
                "\n[ERROR]: line_data should have dimensions (M, N), where "
                "N >= 8. Received {}\n".format(lines.shape)
            ))

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

        for line in lines:
            coords_x.extend([line[1], line[5], None])
            coords_y.extend([line[2], line[6], None])
            coords_z.extend([line[3], line[7], None])

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

        self._fig.add_trace(trace, row = row, col = col)


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
            The row of the subplot to add the traces to. The default is 1.
        col : int, optional
            The column of the subplot to add the traces to. The default is 1.

        '''

        # Add precomputed traces
        self._fig.add_traces(
            traces,
            rows = [row] * len(traces),
            cols = [col] * len(traces)
        )


    def show(self):
        '''Show the Plotly figure.

        Note that the figure will be shown on the Plotly-configured renderer
        (e.g. browser, or PDF). The available renderers can be found by running
        the following code:

        >>> import plotly.io as pio
        >>> pio.renderers

        If you want an interactive figure in the browser, run the following:

        >>> pio.renderers.default = "browser"

        '''

        self._fig.show()


