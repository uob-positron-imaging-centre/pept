#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : grids.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 19.11.2021


import  time
import  textwrap

import  numpy           as      np
import  konigcell       as      kc

from    pept.base       import  Reducer
from    pept            import  PointData, Voxels, Pixels
from    pept.tracking   import  Stack

# PyVista plotting is optional
try:
    import  pyvista     as      pv
except ImportError:
    pass

import  plotly.figure_factory as ff




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
    diameter : float
        The diameter of the imaged tracer.

    column : str or int
        The `PointData` column used to compute the probability distribution,
        given as a name (`str`) or index (`int`).

    dimensions : str or list[int], default "xy"
        The tracer coordinates used to rasterize its trajectory, given as a
        string (e.g. "xy" projects the points onto the XY plane) or a list with
        two column indices (e.g. [1, 3] for XZ).

    resolution : tuple[int, int], default (512, 512)
        The number of pixels used for the rasterization grid in the X and Y
        dimensions.

    xlim : tuple[float, float], optional
        The physical limits in the X dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    ylim : tuple[float, float], optional
        The physical limits in the y dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    max_workers : int, optional
        The maximum number of workers (threads, processes or ranks) to use by
        the parallel executor; if 1, it is sequential (and produces the
        clearest error messages should they happen). If unset, the
        ``os.cpu_count()`` is used.

    verbose : bool or str default True
        If True, time the computation and print the state of the execution.

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
        verbose = True,
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
        self.verbose = verbose


    def fit(self, samples):
        # Reduce / stack list of samples onto a single PointData / array
        samples = Stack().fit(samples)

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
            verbose = self.verbose,
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
    diameter : float
        The diameter of the imaged tracer.

    column : str or int, default "t"
        The `PointData` column used to compute the residence distribution,
        given as a name (`str`) or index (`int`).

    dimensions : str or list[int], default "xy"
        The tracer coordinates used to rasterize its trajectory, given as a
        string (e.g. "xy" projects the points onto the XY plane) or a list with
        two column indices (e.g. [1, 3] for XZ).

    resolution : tuple[int, int], default (512, 512)
        The number of pixels used for the rasterization grid in the X and Y
        dimensions.

    xlim : tuple[float, float], optional
        The physical limits in the X dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    ylim : tuple[float, float], optional
        The physical limits in the y dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    max_workers : int, optional
        The maximum number of workers (threads, processes or ranks) to use by
        the parallel executor; if 1, it is sequential (and produces the
        clearest error messages should they happen). If unset, the
        ``os.cpu_count()`` is used.

    verbose : bool or str default True
        If True, time the computation and print the state of the execution.

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
        verbose = True,
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
        self.verbose = verbose


    def fit(self, samples):
        # Reduce / stack list of samples onto a single PointData / array
        samples = Stack().fit(samples)

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
            verbose = self.verbose,
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
    diameter : float
        The diameter of the imaged tracer.

    column : str or int
        The `PointData` column used to compute the probability distribution,
        given as a name (`str`) or index (`int`).

    dimensions : str or list[int], default "xyz"
        The tracer coordinates used to rasterize its trajectory, given as a
        string (e.g. "xyz" or "zyx") or a list with
        three column indices (e.g. [1, 2, 3] for XYZ).

    resolution : tuple[int, int, int], default (50, 50, 50)
        The number of pixels used for the rasterization grid in the X, Y, Z
        dimensions.

    xlim : tuple[float, float], optional
        The physical limits in the X dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    ylim : tuple[float, float], optional
        The physical limits in the y dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    zlim : tuple[float, float], optional
        The physical limits in the z dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    max_workers : int, optional
        The maximum number of workers (threads, processes or ranks) to use by
        the parallel executor; if 1, it is sequential (and produces the
        clearest error messages should they happen). If unset, the
        ``os.cpu_count()`` is used.

    verbose : bool or str default True
        If True, time the computation and print the state of the execution.

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
        verbose = True,
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
        self.verbose = verbose


    def fit(self, samples):
        # Reduce / stack list of samples onto a single PointData / array
        samples = Stack().fit(samples)

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
            verbose = self.verbose,
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
    diameter : float
        The diameter of the imaged tracer.

    column : str or int
        The `PointData` column used to compute the probability distribution,
        given as a name (`str`) or index (`int`).

    dimensions : str or list[int], default "xyz"
        The tracer coordinates used to rasterize its trajectory, given as a
        string (e.g. "xyz" or "zyx") or a list with
        three column indices (e.g. [1, 2, 3] for XYZ).

    resolution : tuple[int, int, int], default (50, 50, 50)
        The number of pixels used for the rasterization grid in the X, Y, Z
        dimensions.

    xlim : tuple[float, float], optional
        The physical limits in the X dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    ylim : tuple[float, float], optional
        The physical limits in the y dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    zlim : tuple[float, float], optional
        The physical limits in the z dimension of the pixel grid. If unset, it
        is automatically computed to contain all tracer positions (default).

    max_workers : int, optional
        The maximum number of workers (threads, processes or ranks) to use by
        the parallel executor; if 1, it is sequential (and produces the
        clearest error messages should they happen). If unset, the
        ``os.cpu_count()`` is used.

    verbose : bool or str default True
        If True, time the computation and print the state of the execution.

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
        verbose = True,
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
        self.verbose = verbose


    def fit(self, samples):
        # Reduce / stack list of samples onto a single PointData / array
        samples = Stack().fit(samples)

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
            verbose = self.verbose,
        )

        return voxels




class VectorField2D(Reducer):
    '''Compute a 2D vector field - effectively two 2D grids computed from
    two columns, for example X and Y velocities.

    Reducer signature:

    ::

              PointData -> VectorField2D.fit -> VectorGrid2D
        list[PointData] -> VectorField2D.fit -> VectorGrid2D
          numpy.ndarray -> VectorField2D.fit -> VectorGrid2D

    Examples
    --------
    Compute a velocity vector field in the Y and Z dimensions (velocities
    were first calculated using ``pept.tracking.Velocity``):

    >>> from pept.processing import *
    >>> trajectories = pept.PointData(...)
    >>> field = VectorField2D(0.6, ["vy", "vz"], "yz").fit(trajectories)
    >>> field
    VectorGrid2D(xpixels, ypixels)

    Create a quiver plot using Plotly (may be a bit slow):

    >>> scaling = 16
    >>> fig = field.quiver(scaling)
    >>> fig.show()

    Create a 2D vector field (needs PyVista):

    >>> scaling = 16
    >>> fig = field.vectors(scaling)
    >>> fig.plot(cmap = "magma")

    '''

    def __init__(
        self,
        diameter,
        columns = ["vx", "vy"],
        dimensions = "xy",
        resolution = (50, 50),
        xlim = None,
        ylim = None,
        max_workers = None,
        verbose = True,
    ):
        # Type-checking inputs
        self.dimensions = [None, None]
        if isinstance(dimensions, str):
            d = str(dimensions)
            if len(d) != 2 or d[0] not in "xyz" or d[1] not in "xyz":
                raise ValueError(textwrap.fill((
                    "The input `dimensions`, if given as a str, must have "
                    "exactly two characters containing 'x' or 'y'; "
                    f"e.g. 'xy', 'zy'. Received `{d}`."
                )))

            # Transform x -> 1, y -> 2, z -> 3 using ASCII integer `ord`er
            self.dimensions[0] = ord(d[0]) - ord('x') + 1
            self.dimensions[1] = ord(d[1]) - ord('x') + 1
        else:
            d = np.asarray(dimensions, dtype = int)
            if d.ndim != 1 or len(d) != 2:
                raise ValueError(textwrap.fill((
                    "The input `dimensions`, if given as a list, must contain "
                    "exactly two integers representing the column indices "
                    "to use; e.g. `[1, 2]` for xy, `[3, 1]` for zx. "
                    f"Received `{d}`."
                )))
            self.dimensions[0] = d[0]
            self.dimensions[1] = d[1]

        self.diameter = float(diameter)

        columns = list(columns)
        if len(columns) != 2:
            raise ValueError(textwrap.fill((
                "The input `columns` must be a list-like with exactly two "
                "elements, either column names (str, e.g. 'vx') or column "
                f"indices (int, e.g. 4). Received {columns}."
            )))

        self.columns = [None, None]
        for i, col in enumerate(columns):
            if isinstance(col, str):
                self.columns[i] = str(col)
            else:
                self.columns[i] = int(col)

        self.resolution = resolution
        self.xlim = xlim
        self.ylim = ylim

        self.max_workers = max_workers
        self.verbose = verbose


    def fit(self, samples):
        if self.verbose:
            start = time.time()

        # Reduce / stack list of samples onto a single PointData / array
        samples = Stack().fit(samples)

        if not isinstance(samples, PointData):
            samples = PointData(samples)

        grids = [None, None]
        for i, col in enumerate(self.columns):
            if self.verbose:
                print(f"Step {i + 1} / {len(self.columns)}:")

            pixelizer = DynamicProbability2D(
                self.diameter,
                col,
                self.dimensions,
                self.resolution,
                self.xlim,
                self.ylim,
                max_workers = self.max_workers,
                verbose = self.verbose,
            )
            grids[i] = pixelizer.fit(samples)

        if self.verbose:
            end = time.time()
            print(f"Compute 3D vector field in {end - start:4.4f} s.")

        return VectorGrid2D(*grids)




class VectorGrid2D:
    '''Object produced by ``VectorField2D`` storing 2 grids of voxels
    `xpixels`, `ypixels`, for example velocity vector fields / quiver plots.

    Examples
    --------
    Compute a velocity vector field in the Y and Z dimensions (velocities
    were first calculated using ``pept.tracking.Velocity``):

    >>> from pept.processing import *
    >>> trajectories = pept.PointData(...)
    >>> field = VectorField2D(0.6, ["vy", "vz"], "yz").fit(trajectories)
    >>> field
    VectorGrid2D(xpixels, ypixels)

    Create a quiver plot using Plotly (may be a bit slow):

    >>> scaling = 16
    >>> fig = field.quiver(scaling)
    >>> fig.show()

    Create a 2D vector field (needs PyVista):

    >>> scaling = 16
    >>> fig = field.vectors(scaling)
    >>> fig.plot(cmap = "magma")

    '''

    def __init__(
        self,
        xpixels: Pixels,
        ypixels: Pixels,
    ):
        self.xpixels = xpixels
        self.ypixels = ypixels


    def vectors(self, factor = 1):
        # You need to install PyVista to use this function!
        grid = pv.UniformGrid()

        shape = self.xpixels.pixels.shape
        grid.dimensions = [shape[0] + 1, shape[1] + 1, 1]
        grid.origin = list(self.xpixels.lower) + [0]
        grid.spacing = list(self.xpixels.pixel_size) + [0]
        grid.cell_data.values = self.xpixels.pixels.flatten(order="F")

        vectors = grid.cell_centers()

        vectors["vectors"] = np.vstack((
            self.xpixels.pixels.flatten(order="F"),
            self.ypixels.pixels.flatten(order="F"),
            np.zeros(len(self.xpixels.pixels.flatten())),
        )).T

        vectors.active_vectors_name = 'vectors'
        return vectors.glyph(factor = factor)


    def quiver(self, factor = 1):
        xg, yg = self.xpixels.pixel_grids
        xg = 0.5 * (xg[1:] + xg[:-1])
        yg = 0.5 * (yg[1:] + yg[:-1])

        x, y = np.meshgrid(xg, yg)
        fig = ff.create_quiver(
            x,
            y,
            self.xpixels.pixels.T,
            self.ypixels.pixels.T,
            scale = factor,
        )
        return fig


    def __repr__(self):
        return "VectorGrid2D(xpixels, ypixels)"




class VectorField3D(Reducer):
    '''Compute a 3D vector field - effectively three 3D grids computed from
    three columns, for example X, Y and Z velocities.

    Reducer signature:

    ::

              PointData -> VectorField3D.fit -> VectorGrid3D
        list[PointData] -> VectorField3D.fit -> VectorGrid3D
          numpy.ndarray -> VectorField3D.fit -> VectorGrid3D

    Examples
    --------
    Compute a 3D velocity vector field (velocities were first calculated using
    ``pept.tracking.Velocity``):

    >>> from pept.processing import *
    >>> trajectories = pept.PointData(...)
    >>> field = VectorField3D(0.6).fit(trajectories)
    >>> field
    VectorGrid3D(xvoxels, yvoxels, zvoxels)

    Create a 3D vector field (needs PyVista):

    >>> scaling = 16
    >>> fig = field.vectors(scaling)
    >>> fig.plot(cmap = "magma")

    '''

    def __init__(
        self,
        diameter,
        columns = ["vx", "vy", "vz"],
        dimensions = "xyz",
        resolution = (50, 50, 50),
        xlim = None,
        ylim = None,
        zlim = None,
        max_workers = None,
        verbose = True,
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

        columns = list(columns)
        if len(columns) != 3:
            raise ValueError(textwrap.fill((
                "The input `columns` must be a list-like with exactly 3 "
                "elements, either column names (str, e.g. 'vx') or column "
                f"indices (int, e.g. 4). Received {columns}."
            )))

        self.columns = [None, None, None]
        for i, col in enumerate(columns):
            if isinstance(col, str):
                self.columns[i] = str(col)
            else:
                self.columns[i] = int(col)

        self.resolution = resolution
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self.max_workers = max_workers
        self.verbose = verbose


    def fit(self, samples):

        if self.verbose:
            start = time.time()

        # Reduce / stack list of samples onto a single PointData / array
        samples = Stack().fit(samples)

        if not isinstance(samples, PointData):
            samples = PointData(samples)

        grids = [None, None, None]
        for i, col in enumerate(self.columns):
            if self.verbose:
                print(f"Step {i + 1} / {len(self.columns)}...")

            voxelizer = DynamicProbability3D(
                self.diameter,
                col,
                self.dimensions,
                self.resolution,
                self.xlim,
                self.ylim,
                self.zlim,
                max_workers = self.max_workers,
                verbose = self.verbose,
            )
            grids[i] = voxelizer.fit(samples)

        if self.verbose:
            end = time.time()
            print(f"Compute 3D vector field in {end - start:4.4f} s.")

        return VectorGrid3D(*grids)




class VectorGrid3D:
    '''Object produced by ``VectorField3D`` storing 3 grids of voxels
    `xvoxels`, `yvoxels`, `zvoxels`, for example velocity vector fields /
    quiver plots.

    Examples
    --------
    Compute a 3D velocity vector field (velocities were first calculated using
    ``pept.tracking.Velocity``):

    >>> from pept.processing import *
    >>> trajectories = pept.PointData(...)
    >>> field = VectorField3D(0.6).fit(trajectories)
    >>> field
    VectorGrid3D(xvoxels, yvoxels, zvoxels)

    Create a 3D vector field (needs PyVista):

    >>> scaling = 16
    >>> fig = field.vectors(scaling)
    >>> fig.plot(cmap = "magma")

    '''

    def __init__(
        self,
        xvoxels: Voxels,
        yvoxels: Voxels,
        zvoxels: Voxels,
    ):
        self.xvoxels = xvoxels
        self.yvoxels = yvoxels
        self.zvoxels = zvoxels


    def vectors(self, factor = 1):
        # You need to install PyVista to use this function!
        grid = pv.UniformGrid()
        grid.dimensions = np.array(self.xvoxels.voxels.shape) + 1
        grid.origin = self.xvoxels.lower
        grid.spacing = self.xvoxels.voxel_size
        grid.cell_data.values = self.xvoxels.voxels.flatten(order="F")

        vectors = grid.cell_centers()

        vectors["vectors"] = np.vstack((
            self.xvoxels.voxels.flatten(order="F"),
            self.yvoxels.voxels.flatten(order="F"),
            self.zvoxels.voxels.flatten(order="F"),
        )).T

        vectors.active_vectors_name = 'vectors'
        return vectors.glyph(factor = factor)


    def __repr__(self):
        return "VectorGrid3D(xvoxels, yvoxels, zvoxels)"
