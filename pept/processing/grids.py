#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : grids.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 19.11.2021


import  textwrap

import  numpy           as      np
import  konigcell       as      kc

from    pept.base       import  Reducer
from    pept            import  PointData
from    pept.tracking   import  Stack




class DynamicProbability2D(Reducer):
    '''Compute the 2D probability distribution of some tracer quantity (eg
    velocity in each cell).

    Reducer signature:

    ::

              PointData -> DynamicProbability2D.fit -> Pixels
        list[PointData] -> DynamicProbability2D.fit -> Pixels
          numpy.ndarray -> DynamicProbability2D.fit -> Pixels

    This reducer calculates the average value of the tracer quantity in each
    cell of a 2D pixel grid; it uses the full projected tracer area for the
    pixelization step, so the distribution is accurate for arbitrarily fine
    resolutions.

    Parameters
    ----------
    diameter: float
        The diameter of the imaged tracer.

    column: str or int
        The `PointData` column used to compute the probability distribution,
        given as a name (`str`) or index (`int`).

    dimensions: str or list[int], default "xy"
        The tracer coordinates used to rasterize its trajectory, given as a
        string (e.g. "xy" projects the points onto the XY plane) or a list with
        two column indices (e.g. [1, 3] for XZ).

    resolution: tuple[int, int], default (512, 512)
        The number of pixels used for the rasterization grid in the X and Y
        dimensions.

    xlim: tuple[float, float], optional
        The physical limits in the X dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    ylim: tuple[float, float], optional
        The physical limits in the y dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    max_workers: int, optional
        The maximum number of threads to use for computing the probability
        grid.

    Examples
    --------
    Compute the velocity probability distribution of a single tracer trajectory
    having a column named "v" corresponding to the tracer velocity:

    >>> trajectories = pept.load(...)
    >>> pixels_vel = DynamicProbability2D(1.2, "v", "xy").fit(trajectories)

    Plot the pixel grid:

    >>> from pept.plots import PlotlyGrapher2D
    >>> PlotlyGrapher2D().add_pixels(pixels_vel).show()

    For multiple tracer trajectories, you can use ``Segregate`` then
    ``SplitAll('label')`` before calling this reducer to rasterize each
    trajectory separately:

    >>> vel_pipeline = pept.Pipeline([
    >>>     Segregate(20, 10),
    >>>     SplitAll("label"),
    >>>     DynamicProbability2D(1.2, "v", "xy")
    >>> ])
    >>> pixels_vel = vel_pipeline.fit(trajectories)

    '''

    def __init__(
        self,
        diameter,
        column,
        dimensions = "xy",
        resolution = (512, 512),
        xlim = None,
        ylim = None,
        max_workers = None,
    ):
        # Type-checking inputs
        self.dimensions = [None, None]
        if isinstance(dimensions, str):
            d = str(dimensions)
            if len(d) != 2 or d[0] not in "xyz" or d[1] not in "xyz":
                raise ValueError(textwrap.fill((
                    "The input `dimensions`, if given as a str, must have "
                    "exactly two characters containing 'x', 'y', or 'z'; e.g. "
                    f"'xy', 'zy'. Received `{d}`."
                )))

            # Transform x -> 1, y -> 2, z -> 3 using ASCII integer `ord`er
            self.dimensions[0] = ord(d[0]) - ord('x') + 1
            self.dimensions[1] = ord(d[1]) - ord('x') + 1
        else:
            d = np.asarray(dimensions, dtype = int)
            if d.ndim != 1 or len(d) != 2:
                raise ValueError(textwrap.fill((
                    "The input `dimensions`, if given as a list, must contain "
                    "exactly two integers representing the column indices to "
                    "use; e.g. `[1, 2]` for xy, `[3, 2]` for yz. Received "
                    f"`{d}`."
                )))
            self.dimensions[0] = d[0]
            self.dimensions[1] = d[1]

        self.diameter = float(diameter)
        if isinstance(column, str):
            self.column = str(column)
        else:
            self.column = int(column)

        self.resolution = resolution
        self.xlim = xlim
        self.ylim = ylim

        self.max_workers = max_workers


    def fit(self, samples, verbose = True):
        # Reduce / stack list of samples onto a single PointData / array
        samples = Stack().fit(samples)
        verbose = bool(verbose)

        if not isinstance(samples, PointData):
            samples = PointData(samples)

        # The column whose values we will rasterize is either given as a named
        # column, or an integer index
        if isinstance(self.column, str):
            col_idx = samples.columns.index(self.column)
        else:
            col_idx = self.column

        # konigcell.probability2d will split points for parallel rasterization
        # to minimise memory consumption; need to add NaNs between points
        # samples to separate individual trajectories
        coordinates = []
        values = []
        for sample in samples:
            coordinates.append(sample.points[:, self.dimensions])
            coordinates.append(np.full(2, np.nan))

            values.append(sample.points[:, col_idx])
            values.append(np.full(1, np.nan))

        coordinates = np.vstack(coordinates)
        values = np.hstack(values)

        pixels = kc.dynamic_prob2d(
            coordinates,
            values[:-1],
            self.diameter / 2,
            resolution = self.resolution,
            xlim = self.xlim,
            ylim = self.ylim,
            max_workers = self.max_workers,
            verbose = verbose,
        )

        return pixels




class ResidenceDistribution2D(Reducer):
    '''Compute the 2D residence distribution of some tracer quantity (eg
    time spent in each cell).

    Reducer signature:

    ::

              PointData -> ResidenceDistribution2D.fit -> Pixels
        list[PointData] -> ResidenceDistribution2D.fit -> Pixels
          numpy.ndarray -> ResidenceDistribution2D.fit -> Pixels

    This reducer calculates the cumulative value of the tracer quantity in each
    cell of a 2D pixel grid; it uses the full projected tracer area for the
    pixelization step, so the distribution is accurate for arbitrarily fine
    resolutions.

    Parameters
    ----------
    diameter: float
        The diameter of the imaged tracer.

    column: str or int, default "t"
        The `PointData` column used to compute the residence distribution,
        given as a name (`str`) or index (`int`).

    dimensions: str or list[int], default "xy"
        The tracer coordinates used to rasterize its trajectory, given as a
        string (e.g. "xy" projects the points onto the XY plane) or a list with
        two column indices (e.g. [1, 3] for XZ).

    resolution: tuple[int, int], default (512, 512)
        The number of pixels used for the rasterization grid in the X and Y
        dimensions.

    xlim: tuple[float, float], optional
        The physical limits in the X dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    ylim: tuple[float, float], optional
        The physical limits in the y dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    max_workers: int, optional
        The maximum number of threads to use for computing the probability
        grid.

    Examples
    --------
    Compute the residence time distribution of a single tracer trajectory:

    >>> trajectories = pept.load(...)
    >>> pixels_rtd = ResidenceDistribution2D(1.2, "t", "xy").fit(trajectories)

    Plot the pixel grid:

    >>> from pept.plots import PlotlyGrapher2D
    >>> PlotlyGrapher2D().add_pixels(pixels_rtd).show()

    For multiple tracer trajectories, you can use ``Segregate`` then
    ``SplitAll('label')`` before calling this reducer to rasterize each
    trajectory separately:

    >>> rtd_pipeline = pept.Pipeline([
    >>>     Segregate(20, 10),
    >>>     SplitAll("label"),
    >>>     ResidenceDistribution2D(1.2, "t", "xy")
    >>> ])
    >>> pixels_rtd = rtd_pipeline.fit(trajectories)

    '''

    def __init__(
        self,
        diameter,
        column = "t",
        dimensions = "xy",
        resolution = (512, 512),
        xlim = None,
        ylim = None,
        max_workers = None,
    ):
        # Type-checking inputs
        self.dimensions = [None, None]
        if isinstance(dimensions, str):
            d = str(dimensions)
            if len(d) != 2 or d[0] not in "xyz" or d[1] not in "xyz":
                raise ValueError(textwrap.fill((
                    "The input `dimensions`, if given as a str, must have "
                    "exactly two characters containing 'x', 'y', or 'z'; e.g. "
                    f"'xy', 'zy'. Received `{d}`."
                )))

            # Transform x -> 1, y -> 2, z -> 3 using ASCII integer `ord`er
            self.dimensions[0] = ord(d[0]) - ord('x') + 1
            self.dimensions[1] = ord(d[1]) - ord('x') + 1
        else:
            d = np.asarray(dimensions, dtype = int)
            if d.ndim != 1 or len(d) != 2:
                raise ValueError(textwrap.fill((
                    "The input `dimensions`, if given as a list, must contain "
                    "exactly two integers representing the column indices to "
                    "use; e.g. `[1, 2]` for xy, `[3, 2]` for yz. Received "
                    f"`{d}`."
                )))
            self.dimensions[0] = d[0]
            self.dimensions[1] = d[1]

        self.diameter = float(diameter)
        if isinstance(column, str):
            self.column = str(column)
        else:
            self.column = int(column)

        self.resolution = resolution
        self.xlim = xlim
        self.ylim = ylim

        self.max_workers = max_workers


    def fit(self, samples, verbose = True):
        # Reduce / stack list of samples onto a single PointData / array
        samples = Stack().fit(samples)
        verbose = bool(verbose)

        if not isinstance(samples, PointData):
            samples = PointData(samples)

        # The column whose values we will rasterize is either given as a named
        # column, or an integer index
        if isinstance(self.column, str):
            col_idx = samples.columns.index(self.column)
        else:
            col_idx = self.column

        # konigcell.probability2d will split points for parallel rasterization
        # to minimise memory consumption; need to add NaNs between points
        # samples to separate individual trajectories
        coordinates = []
        values = []
        for sample in samples:
            coordinates.append(sample.points[:, self.dimensions])
            coordinates.append(np.full(2, np.nan))

            values.append(sample.points[:, col_idx])
            values.append(np.full(1, np.nan))

        coordinates = np.vstack(coordinates)
        values = np.hstack(values)
        values_dt = values[1:] - values[:-1]

        pixels = kc.dynamic2d(
            coordinates,
            kc.RATIO,
            values_dt,
            self.diameter / 2,
            resolution = self.resolution,
            xlim = self.xlim,
            ylim = self.ylim,
            max_workers = self.max_workers,
            verbose = verbose,
        )

        return pixels




class DynamicProbability3D(Reducer):
    '''Compute the 3D probability distribution of some tracer quantity (eg
    velocity in each cell).

    Reducer signature:

    ::

              PointData -> DynamicProbability3D.fit -> Voxels
        list[PointData] -> DynamicProbability3D.fit -> Voxels
          numpy.ndarray -> DynamicProbability3D.fit -> Voxels

    This reducer calculates the average value of the tracer quantity in each
    cell of a 3D voxel grid; it uses the full projected tracer area for the
    voxelization step, so the distribution is accurate for arbitrarily fine
    resolutions.

    Parameters
    ----------
    diameter: float
        The diameter of the imaged tracer.

    column: str or int
        The `PointData` column used to compute the probability distribution,
        given as a name (`str`) or index (`int`).

    dimensions: str or list[int], default "xyz"
        The tracer coordinates used to rasterize its trajectory, given as a
        string (e.g. "xyz" or "zyx") or a list with
        three column indices (e.g. [1, 2, 3] for XYZ).

    resolution: tuple[int, int, int], default (50, 50, 50)
        The number of pixels used for the rasterization grid in the X, Y, Z
        dimensions.

    xlim: tuple[float, float], optional
        The physical limits in the X dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    ylim: tuple[float, float], optional
        The physical limits in the y dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    zlim: tuple[float, float], optional
        The physical limits in the z dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    max_workers: int, optional
        The maximum number of threads to use for computing the probability
        grid.

    Examples
    --------
    Compute the velocity probability distribution of a single tracer trajectory
    having a column named "v" corresponding to the tracer velocity:

    >>> trajectories = pept.load(...)
    >>> voxels_vel = DynamicProbability3D(1.2, "v").fit(trajectories)

    Plot the pixel grid:

    >>> from pept.plots import PlotlyGrapher
    >>> PlotlyGrapher().add_voxels(voxels_vel).show()

    For multiple tracer trajectories, you can use ``Segregate`` then
    ``SplitAll('label')`` before calling this reducer to rasterize each
    trajectory separately:

    >>> vel_pipeline = pept.Pipeline([
    >>>     Segregate(20, 10),
    >>>     SplitAll("label"),
    >>>     DynamicProbability3D(1.2, "v")
    >>> ])
    >>> voxels_vel = vel_pipeline.fit(trajectories)

    '''

    def __init__(
        self,
        diameter,
        column,
        dimensions = "xyz",
        resolution = (50, 50, 50),
        xlim = None,
        ylim = None,
        zlim = None,
        max_workers = None,
    ):
        # Type-checking inputs
        self.dimensions = [None, None, None]
        if isinstance(dimensions, str):
            d = str(dimensions)
            if len(d) != 3 or d[0] not in "xyz" or d[1] not in "xyz" or \
                    d[2] not in "xyz":
                raise ValueError(textwrap.fill((
                    "The input `dimensions`, if given as a str, must have "
                    "exactly three characters containing 'x', 'y', or 'z'; "
                    f"e.g. 'xyz', 'zxy'. Received `{d}`."
                )))

            # Transform x -> 1, y -> 2, z -> 3 using ASCII integer `ord`er
            self.dimensions[0] = ord(d[0]) - ord('x') + 1
            self.dimensions[1] = ord(d[1]) - ord('x') + 1
            self.dimensions[2] = ord(d[2]) - ord('x') + 1
        else:
            d = np.asarray(dimensions, dtype = int)
            if d.ndim != 1 or len(d) != 3:
                raise ValueError(textwrap.fill((
                    "The input `dimensions`, if given as a list, must contain "
                    "exactly three integers representing the column indices "
                    "to use; e.g. `[1, 2, 3]` for xyz, `[3, 1, 2]` for zxy. "
                    f"Received `{d}`."
                )))
            self.dimensions[0] = d[0]
            self.dimensions[1] = d[1]
            self.dimensions[2] = d[2]

        self.diameter = float(diameter)
        if isinstance(column, str):
            self.column = str(column)
        else:
            self.column = int(column)

        self.resolution = resolution
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self.max_workers = max_workers


    def fit(self, samples, verbose = True):
        # Reduce / stack list of samples onto a single PointData / array
        samples = Stack().fit(samples)
        verbose = bool(verbose)

        if not isinstance(samples, PointData):
            samples = PointData(samples)

        # The column whose values we will rasterize is either given as a named
        # column, or an integer index
        if isinstance(self.column, str):
            col_idx = samples.columns.index(self.column)
        else:
            col_idx = self.column

        # konigcell.probability2d will split points for parallel rasterization
        # to minimise memory consumption; need to add NaNs between points
        # samples to separate individual trajectories
        coordinates = []
        values = []
        for sample in samples:
            coordinates.append(sample.points[:, self.dimensions])
            coordinates.append(np.full(3, np.nan))

            values.append(sample.points[:, col_idx])
            values.append(np.full(1, np.nan))

        coordinates = np.vstack(coordinates)
        values = np.hstack(values)

        voxels = kc.dynamic_prob3d(
            coordinates,
            values[:-1],
            self.diameter / 2,
            resolution = self.resolution,
            xlim = self.xlim,
            ylim = self.ylim,
            zlim = self.zlim,
            max_workers = self.max_workers,
            verbose = verbose,
        )

        return voxels




class ResidenceDistribution3D(Reducer):
    '''Compute the 3D residence distribution of some tracer quantity (eg
    time spent in each cell).

    Reducer signature:

    ::

              PointData -> ResidenceDistribution3D.fit -> Pixels
        list[PointData] -> ResidenceDistribution3D.fit -> Pixels
          numpy.ndarray -> ResidenceDistribution3D.fit -> Pixels

    This reducer calculates the cumulative value of the tracer quantity in each
    cell of a 3D voxel grid; it uses the full projected tracer area for the
    voxelization step, so the distribution is accurate for arbitrarily fine
    resolutions.

    Parameters
    ----------
    diameter: float
        The diameter of the imaged tracer.

    column: str or int
        The `PointData` column used to compute the probability distribution,
        given as a name (`str`) or index (`int`).

    dimensions: str or list[int], default "xyz"
        The tracer coordinates used to rasterize its trajectory, given as a
        string (e.g. "xyz" or "zyx") or a list with
        three column indices (e.g. [1, 2, 3] for XYZ).

    resolution: tuple[int, int, int], default (50, 50, 50)
        The number of pixels used for the rasterization grid in the X, Y, Z
        dimensions.

    xlim: tuple[float, float], optional
        The physical limits in the X dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    ylim: tuple[float, float], optional
        The physical limits in the y dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    zlim: tuple[float, float], optional
        The physical limits in the z dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    max_workers: int, optional
        The maximum number of threads to use for computing the probability
        grid.

    Examples
    --------
    Compute the residence time distribution of a single tracer trajectory:

    >>> trajectories = pept.load(...)
    >>> voxels_rtd = ResidenceDistribution3D(1.2, "t").fit(trajectories)

    Plot the pixel grid:

    >>> from pept.plots import PlotlyGrapher
    >>> PlotlyGrapher().add_voxels(voxels_rtd).show()

    For multiple tracer trajectories, you can use ``Segregate`` then
    ``SplitAll('label')`` before calling this reducer to rasterize each
    trajectory separately:

    >>> rtd_pipeline = pept.Pipeline([
    >>>     Segregate(20, 10),
    >>>     SplitAll("label"),
    >>>     ResidenceDistribution3D(1.2, "t")
    >>> ])
    >>> voxels_rtd = rtd_pipeline.fit(trajectories)

    '''


    def __init__(
        self,
        diameter,
        column = "t",
        dimensions = "xyz",
        resolution = (50, 50, 50),
        xlim = None,
        ylim = None,
        zlim = None,
        max_workers = None,
    ):
        # Type-checking inputs
        self.dimensions = [None, None, None]
        if isinstance(dimensions, str):
            d = str(dimensions)
            if len(d) != 3 or d[0] not in "xyz" or d[1] not in "xyz" or \
                    d[2] not in "xyz":
                raise ValueError(textwrap.fill((
                    "The input `dimensions`, if given as a str, must have "
                    "exactly three characters containing 'x', 'y', or 'z'; "
                    f"e.g. 'xyz', 'zxy'. Received `{d}`."
                )))

            # Transform x -> 1, y -> 2, z -> 3 using ASCII integer `ord`er
            self.dimensions[0] = ord(d[0]) - ord('x') + 1
            self.dimensions[1] = ord(d[1]) - ord('x') + 1
            self.dimensions[2] = ord(d[2]) - ord('x') + 1
        else:
            d = np.asarray(dimensions, dtype = int)
            if d.ndim != 1 or len(d) != 3:
                raise ValueError(textwrap.fill((
                    "The input `dimensions`, if given as a list, must contain "
                    "exactly three integers representing the column indices "
                    "to use; e.g. `[1, 2, 3]` for xyz, `[3, 1, 2]` for zxy. "
                    f"Received `{d}`."
                )))
            self.dimensions[0] = d[0]
            self.dimensions[1] = d[1]
            self.dimensions[2] = d[2]

        self.diameter = float(diameter)
        if isinstance(column, str):
            self.column = str(column)
        else:
            self.column = int(column)

        self.resolution = resolution
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self.max_workers = max_workers


    def fit(self, samples, verbose = True):
        # Reduce / stack list of samples onto a single PointData / array
        samples = Stack().fit(samples)
        verbose = bool(verbose)

        if not isinstance(samples, PointData):
            samples = PointData(samples)

        # The column whose values we will rasterize is either given as a named
        # column, or an integer index
        if isinstance(self.column, str):
            col_idx = samples.columns.index(self.column)
        else:
            col_idx = self.column

        # konigcell.probability2d will split points for parallel rasterization
        # to minimise memory consumption; need to add NaNs between points
        # samples to separate individual trajectories
        coordinates = []
        values = []
        for sample in samples:
            coordinates.append(sample.points[:, self.dimensions])
            coordinates.append(np.full(3, np.nan))

            values.append(sample.points[:, col_idx])
            values.append(np.full(1, np.nan))

        coordinates = np.vstack(coordinates)
        values = np.hstack(values)
        values_dt = values[1:] - values[:-1]

        voxels = kc.dynamic3d(
            coordinates,
            kc.RATIO,
            values_dt,
            self.diameter / 2,
            resolution = self.resolution,
            xlim = self.xlim,
            ylim = self.ylim,
            zlim = self.zlim,
            max_workers = self.max_workers,
            verbose = verbose,
        )

        return voxels
