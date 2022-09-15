#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mixing.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 24.08.2022


import  os
import  textwrap


import  numpy               as      np
import  scipy

from    tqdm                import  tqdm

from    pept                import  PointData
from    pept.base           import  Reducer
from    pept.tracking       import  Stack
from    pept                import  plots

from    plotly.subplots     import  make_subplots
import  plotly.graph_objs   as      go




def enlarge(lim):
    '''Correctly enlarge two limits, e.g. [xmin, xmax]
    '''
    eps = np.finfo(float).resolution
    lim = lim.copy()

    if lim[0] < 0:
        lim[0] = lim[0] * 1.0001 - eps
    else:
        lim[0] = lim[0] * 0.9999 - eps

    if lim[1] < 0:
        lim[1] = lim[1] * 0.9999 + eps
    else:
        lim[1] = lim[1] * 1.0001 + eps

    return lim




def _check_point3d(**kwargs):
    # Extract first argument
    for name, point in kwargs.items():
        break

    if point.ndim != 1 or len(point) != 3:
        raise ValueError(textwrap.fill((
            f"The input point `{name}` should be a vector-like with exactly 3 "
            f"elements. Received `{point}`."
        )))




class LaceyColors(Reducer):
    '''Compute Lacey-like mixing image, with streamlines passing through plane
    1 being split into Red and Blue tracers, then evaluated into RGB pixels at
    a later point in plane 2.

    Intuitively, red and blue pixels will contain unmixed streamlines, while
    purple pixels will indicate mixing.

    Reducer signature:

    ::

               PointData -> LaceyColors.fit -> (height, width, 3) np.ndarray
         list[PointData] -> LaceyColors.fit -> (height, width, 3) np.ndarray
        list[np.ndarray] -> LaceyColors.fit -> (height, width, 3) np.ndarray

    **Each sample in the input `PointData` is treated as a separate streamline
    / tracer pass. You can group passes using `Segregate + GroupBy("label")`**.

    The first plane where tracers are split into Red and Blue streamlines is
    defined by a point `p1` and direction axis `ax1`. **The point `p1` should
    be the middle of the pipe**.

    The second plane where mixing is evaluated is similarly defined by `p2` and
    `ax2`. **The point `p2` should be the middle of the volume / pipe**.

    If the direction vectors `ax1` and `ax2` are undefined (`None`), the
    tracers are assumed to follow a straight line between `p1` and `p2`.

    The `max_distance` parameter defines the maximum distance allowed between
    a point and a plane to be considered part of it. The `resolution` defines
    the number of pixels in the height and width of the resulting image.

    Examples
    --------
    Consider a pipe-flow experiment, with tracers moving from side to side in
    multiple passes / streamlines. First locate the tracers, then split their
    trajectories into each individual pass:

    >>> import pept
    >>> from pept.tracking import *
    >>>
    >>> split_pipe = pept.Pipeline([
    >>>     Segregate(window = 10, max_distance = 20),  # Appends label column
    >>>     GroupBy("label"),                           # Splits into samples
    >>>     Reorient(),                                 # Align with X axis
    >>>     Center(),                                   # Center points at 0
    >>>     Stack(),
    >>> ])
    >>> streamlines = split_pipe.fit(trajectories)

    Now each sample in `streamlines` corresponds to a single tracer pass, e.g.
    `streamlines[0]` is the first pass, `streamlines[1]` is the second. The
    passes were reoriented and centred such that the pipe is aligned with the
    X axis.

    Now the `LaceyColors` reducer can be used to create an image of the mixing
    between the pipe entrance and exit:

    >>> from pept.processing import LaceyColors
    >>> entrance = [-100, 0, 0]     # Pipe data was aligned with X and centred
    >>> exit = [100, 0, 0]
    >>> lacey_image = LaceyColors(entrance, exit).fit(streamlines)
    >>> print(lacey_image.shape)    # RGB channels of image
    (8, 8, 3)

    Now the image can be visualised e.g. with Plotly:

    >>> from pept.plots import PlotlyGrapher2D
    >>> PlotlyGrapher2D().add_image(lacey_image).show()
    '''

    def __init__(
        self,
        p1,
        p2,
        ax1 = None,
        ax2 = None,
        max_distance = 10,
        resolution = (8, 8),
    ):
        self.p1 = np.asarray(p1, dtype = float)
        _check_point3d(p1 = self.p1)

        if ax1 is not None:
            ax1 = np.asarray(ax1, dtype = float)
            self.ax1 = ax1 / np.linalg.norm(ax1)
            _check_point3d(ax1 = self.ax1)

        self.p2 = np.asarray(p2, dtype = float)
        _check_point3d(p2 = self.p2)

        if ax2 is not None:
            ax2 = np.asarray(ax2, dtype = float)
            self.ax2 = ax2 / np.linalg.norm(ax2)
            _check_point3d(ax2 = self.ax2)

        self.max_distance = float(max_distance)
        self.resolution = tuple(resolution)


    def fit(self, trajectories):

        trajectories = Stack().fit(trajectories)
        if not isinstance(trajectories, PointData):
            raise TypeError(textwrap.fill((
                "The input `trajectories` must be pept.PointData-like. "
                f"Received `{type(trajectories)}`."
            )))

        if self.ax1 is None:
            self.ax1 = self.p2 - self.p1

        if self.ax2 is None:
            self.ax2 = self.p2 - self.p1

        self.ax1 = self.ax1 / np.linalg.norm(self.ax1)
        self.ax2 = self.ax2 / np.linalg.norm(self.ax2)

        # Aliases
        p1 = self.p1
        ax1 = self.ax1
        p2 = self.p2
        ax2 = self.ax2

        max_distance = self.max_distance
        resolution = self.resolution

        # For each pass, find the closest point to the start and end planes
        points_start = []
        points_end = []

        for traj in trajectories:
            xyz = traj.points[:, 1:4]
            xyz = xyz[np.isfinite(xyz).all(axis = 1)]

            dists_start = np.abs(np.dot(xyz - p1, ax1))
            dists_end = np.abs(np.dot(xyz - p2, ax2))

            min_dist_start = np.min(dists_start)
            min_dist_end = np.min(dists_end)

            if min_dist_start < max_distance and min_dist_end < max_distance:
                points_start.append(xyz[np.argmin(dists_start)])
                points_end.append(xyz[np.argmin(dists_end)])

        points_start = np.array(points_start)
        points_end = np.array(points_end)

        # Split starting points into two species - Red and Blue - along the PCA
        points_start_centred = points_start - p1

        cov = np.cov(points_start_centred.T)
        evals, evecs = np.linalg.eig(cov)

        evals_argsorted = evals.argsort()[::-1]
        evals = evals[evals_argsorted]
        evecs = evecs[:, evals_argsorted]

        points_start2d = points_start_centred @ evecs[:, :2]

        # Project ending points onto 2D plane by to the Principal Components
        points_end_centred = points_end - p2

        cov = np.cov(points_end_centred.T)
        evals, evecs = np.linalg.eig(cov)

        evals_argsorted = evals.argsort()[::-1]
        evals = evals[evals_argsorted]
        evecs = evecs[:, evals_argsorted]

        points_end2d = points_end_centred @ evecs[:, :2]

        # Create RGB image with the R and B channels == the number of passes
        ired = points_start2d[:, 0] < 0
        iblu = points_start2d[:, 0] >= 0

        limits = np.c_[points_end2d.min(axis = 0), points_end2d.max(axis = 0)]
        xlim = enlarge(limits[0, :])
        ylim = enlarge(limits[1, :])

        xsize = (xlim[1] - xlim[0]) / resolution[0]
        ysize = (ylim[1] - ylim[0]) / resolution[1]

        red_mix = np.zeros(resolution)
        blu_mix = np.zeros(resolution)

        # Find indices within pixels
        red_ix = ((points_end2d[ired, 0] - xlim[0]) / xsize).astype(int)
        red_iy = ((points_end2d[ired, 1] - ylim[0]) / ysize).astype(int)
        for i in range(len(red_ix)):
            red_mix[red_ix[i], red_iy[i]] += 1

        blu_ix = ((points_end2d[iblu, 0] - xlim[0]) / xsize).astype(int)
        blu_iy = ((points_end2d[iblu, 1] - ylim[0]) / ysize).astype(int)
        for i in range(len(blu_ix)):
            blu_mix[blu_ix[i], blu_iy[i]] += 1

        # Stack RGB channels
        red_mix *= 255 / red_mix.max()
        blu_mix *= 255 / blu_mix.max()

        mix_img = np.zeros((resolution[0], resolution[1], 3), dtype = int)
        mix_img[:, :, 0] = red_mix.astype(int)
        mix_img[:, :, 2] = blu_mix.astype(int)

        return mix_img




class LaceyColorsLinear(Reducer):
    '''Apply the `LaceyColors` mixing algorithm at `num_divisions` equidistant
    points between `p1` and `p2`, saving images at each step in `directory`.

    Reducer signature:

    ::

               PointData -> LaceyColors.fit -> (height, width, 3) np.ndarray
         list[PointData] -> LaceyColors.fit -> (height, width, 3) np.ndarray
        list[np.ndarray] -> LaceyColors.fit -> (height, width, 3) np.ndarray

    For details about the mixing algorithm itself, check the `LaceyColors`
    documentation.

    The generated images (saved in `directory` with `height` x `width` pixels)
    can be stitched into a video using `pept.plots.make_video`.

    Examples
    --------
    Consider a pipe-flow experiment, with tracers moving from side to side in
    multiple passes / streamlines. First locate the tracers, then split their
    trajectories into each individual pass:

    >>> import pept
    >>> from pept.tracking import *
    >>>
    >>> split_pipe = pept.Pipeline([
    >>>     Segregate(window = 10, max_distance = 20),  # Appends label column
    >>>     GroupBy("label"),                           # Splits into samples
    >>>     Reorient(),                                 # Align with X axis
    >>>     Center(),                                   # Center points at 0
    >>>     Stack(),
    >>> ])
    >>> streamlines = split_pipe.fit(trajectories)

    Now each sample in `streamlines` corresponds to a single tracer pass, e.g.
    `streamlines[0]` is the first pass, `streamlines[1]` is the second. The
    passes were reoriented and centred such that the pipe is aligned with the
    X axis.

    Now the `LaceyColorsLinear` reducer can be used to create images of the
    mixing between the pipe entrance and exit:

    >>> from pept.processing import LaceyColorsLinear
    >>> entrance = [-100, 0, 0]     # Pipe data was aligned with X and centred
    >>> exit = [100, 0, 0]
    >>> LaceyColorsLinear(
    >>>     directory = "lacey",    # Creates directory and saves images there
    >>>     p1 = entrance,
    >>>     p2 = exit,
    >>> ).fit(streamlines)

    Now the directory "lacey" was created inside your current working folder,
    and all Lacey images saved there as "frame0000.png", "frame0001.png", etc.
    You can stitch all images together into a video using
    `pept.plots.make_video`:

    >>> import pept
    >>> pept.plots.make_video("lacey/frame*.png", output = "lacey/video.avi")
    '''

    def __init__(
        self,
        directory,
        p1, p2,
        num_divisions = 50,
        max_distance = 10,
        resolution = (8, 8),
        height = 1000,
        width = 1000,
        prefix = "frame",
    ):
        self.directory = directory
        self.num_divisions = int(num_divisions)

        self.p1 = np.asarray(p1, dtype = float)
        self.p2 = np.asarray(p2, dtype = float)

        self.max_distance = float(max_distance)
        self.resolution = tuple(resolution)

        self.height = int(height)
        self.width = int(width)
        self.prefix = prefix


    def fit(self, trajectories, verbose = True):

        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        divisions = np.c_[
            np.linspace(self.p1[0], self.p2[0], self.num_divisions),
            np.linspace(self.p1[1], self.p2[1], self.num_divisions),
            np.linspace(self.p1[2], self.p2[2], self.num_divisions),
        ]

        axis = self.p2 - self.p1

        if verbose:
            divisions = tqdm(divisions, desc = "LaceyColorsLinear :")

        for i, div in enumerate(divisions):
            image = LaceyColors(
                p1 = self.p1,
                p2 = div,
                ax1 = axis,
                ax2 = axis,
                max_distance = self.max_distance,
                resolution = self.resolution,
            ).fit(trajectories)

            travelled = np.linalg.norm(div - self.p1)
            grapher = plots.PlotlyGrapher2D(subplot_titles = [
                f"Travelled {travelled} mm"
            ])
            grapher.add_trace(go.Image(z = image))
            grapher.fig.write_image(
                f"{self.directory}/{self.prefix}{i:0>4}.png",
                height = self.height,
                width = self.width,
            )




class RelativeDeviations(Reducer):
    '''Compute a Lagrangian mixing measure - the changes in tracer distances
    to a point P1 as they pass through an "inlet" plane and another point P2
    when reaching an "outlet" plane.

    A deviation is computed for each tracer trajectory, yielding a range of
    deviations that can e.g be histogrammed (default). Intuitively, mixing is
    stronger if this distribution of deviations is wider.

    Reducer signature:

    ::

        If ``histogram = True`` (default)
               PointData -> LaceyColors.fit -> plotly.graph_objs.Figure
         list[PointData] -> LaceyColors.fit -> plotly.graph_objs.Figure
        list[np.ndarray] -> LaceyColors.fit -> plotly.graph_objs.Figure

        If ``histogram = False`` (return deviations)
               PointData -> LaceyColors.fit -> (N,) np.ndarray
         list[PointData] -> LaceyColors.fit -> (N,) np.ndarray
        list[np.ndarray] -> LaceyColors.fit -> (N,) np.ndarray


    **Each sample in the input `PointData` is treated as a separate streamline
    / tracer pass. You can group passes using `Segregate + GroupBy("label")`**.

    The first plane where the distances from tracers to a point `p1` is defined
    by the point `p1` and direction axis `ax1`. **The point `p1` should be the
    middle of the pipe**.

    The second plane where relative distances are evaluated is similarly
    defined by `p2` and `ax2`. **The point `p2` should be the middle of the
    volume / pipe**.

    If the direction vectors `ax1` and `ax2` are undefined (`None`), the
    tracers are assumed to follow a straight line between `p1` and `p2`.

    The `max_distance` parameter defines the maximum distance allowed between
    a point and a plane to be considered part of it. The `resolution` defines
    the number of pixels in the height and width of the resulting image.

    The following attributes are always set. A Plotly figure is only generated
    and returned if `histogram = True` (default).

    The extra keyword arguments ``**kwargs`` are passed to the histogram
    creation routine `pept.plots.histogram`. You can e.g. set the YAxis limits
    by adding `ylim = [0, 20]`.

    Attributes
    ----------
    points1 : pept.PointData
        The tracer points selected at the inlet around `p1`.

    points2 : pept.PointData
        The tracer points selected at the outlet around `p2`.

    deviations : (N,) np.ndarray
        The vector of tracer deviations for each tracer pass in `points1` and
        `points2`.

    Examples
    --------
    Consider a pipe-flow experiment, with tracers moving from side to side in
    multiple passes / streamlines. First locate the tracers, then split their
    trajectories into each individual pass:

    >>> import pept
    >>> from pept.tracking import *
    >>>
    >>> split_pipe = pept.Pipeline([
    >>>     Segregate(window = 10, max_distance = 20),  # Appends label column
    >>>     GroupBy("label"),                           # Splits into samples
    >>>     Reorient(),                                 # Align with X axis
    >>>     Center(),                                   # Center points at 0
    >>>     Stack(),
    >>> ])
    >>> streamlines = split_pipe.fit(trajectories)

    Now each sample in `streamlines` corresponds to a single tracer pass, e.g.
    `streamlines[0]` is the first pass, `streamlines[1]` is the second. The
    passes were reoriented and centred such that the pipe is aligned with the
    X axis.

    Now the `RelativeDeviations` reducer can be used to create a histogram of
    tracer deviations due to mixing:

    >>> from pept.processing import RelativeDeviations
    >>> entrance = [-100, 0, 0]     # Pipe data was aligned with X and centred
    >>> exit = [100, 0, 0]
    >>> fig = RelativeDeviations(entrance, exit).fit(streamlines)
    >>> fig.show()

    The deviations themselves can be extracted directly for further processing:

    >>> mixing_algorithm = RelativeDeviations(entrance, exit, histogram=False)
    >>> mixing_algorithm.fit(streamlines)

    >>> deviations = mixing_algorithm.deviations
    >>> inlet_points = mixing_algorithm.points1
    >>> outlet_points = mixing_algorithm.points2
    '''

    def __init__(
        self,
        p1,
        p2,
        ax1 = None,
        ax2 = None,
        max_distance = 10,
        histogram = True,
        **kwargs,
    ):
        self.p1 = np.asarray(p1, dtype = float)
        _check_point3d(p1 = self.p1)

        if ax1 is not None:
            ax1 = np.asarray(ax1, dtype = float)
            self.ax1 = ax1 / np.linalg.norm(ax1)
            _check_point3d(ax1 = self.ax1)

        self.p2 = np.asarray(p2, dtype = float)
        _check_point3d(p2 = self.p2)

        if ax2 is not None:
            ax2 = np.asarray(ax2, dtype = float)
            self.ax2 = ax2 / np.linalg.norm(ax2)
            _check_point3d(ax2 = self.ax2)

        self.max_distance = float(max_distance)
        self.histogram = bool(histogram)

        self.kwargs = kwargs

        # Will be set in `fit`
        self.points1 = None
        self.points2 = None
        self.deviations = None


    def fit(self, trajectories):

        trajectories = Stack().fit(trajectories)
        if not isinstance(trajectories, PointData):
            raise TypeError(textwrap.fill((
                "The input `trajectories` must be pept.PointData-like. "
                f"Received `{type(trajectories)}`."
            )))

        if self.ax1 is None:
            self.ax1 = self.p2 - self.p1

        if self.ax2 is None:
            self.ax2 = self.p2 - self.p1

        self.ax1 = self.ax1 / np.linalg.norm(self.ax1)
        self.ax2 = self.ax2 / np.linalg.norm(self.ax2)

        # Aliases
        p1 = self.p1
        ax1 = self.ax1
        p2 = self.p2
        ax2 = self.ax2

        max_distance = self.max_distance

        # For each pass, find the closest point to the start and end planes
        points_start = []
        points_end = []

        for traj in trajectories:
            xyz = traj.points[:, 1:4]
            xyz = xyz[np.isfinite(xyz).all(axis = 1)]

            dists_start = np.abs(np.dot(xyz - p1, ax1))
            dists_end = np.abs(np.dot(xyz - p2, ax2))

            min_dist_start = np.min(dists_start)
            min_dist_end = np.min(dists_end)

            if min_dist_start < max_distance and min_dist_end < max_distance:
                points_start.append(traj.points[np.argmin(dists_start)])
                points_end.append(traj.points[np.argmin(dists_end)])

        # TODO: check empty trajectories
        self.points1 = trajectories.copy(data = points_start)
        self.points2 = trajectories.copy(data = points_end)

        # Distances to P1 and P2 and relative deviations
        d1 = np.linalg.norm(self.points1.points[:, 1:4] - self.p1, axis = 1)
        d2 = np.linalg.norm(self.points2.points[:, 1:4] - self.p2, axis = 1)

        self.deviations = np.abs(d2 - d1)

        # Return histogram of deviations
        if self.histogram:
            return plots.histogram(
                self.deviations,
                xaxis_title = "Relative Deviation (mm)",
                yaxis_title = "Probability (%)",
                **self.kwargs,
            )
        else:
            return self.deviations




class RelativeDeviationsLinear(Reducer):
    '''Apply the `RelativeDeviations` mixing algorithm at `num_divisions`
    equidistant points between `p1` and `p2`, saving histogram images at each
    step in `directory`.

    Reducer signature:

    ::

               PointData -> LaceyColors.fit -> plotly.graph_objs.Figure
         list[PointData] -> LaceyColors.fit -> plotly.graph_objs.Figure
        list[np.ndarray] -> LaceyColors.fit -> plotly.graph_objs.Figure

    For details about the mixing algorithm itself, check the
    `RelativeDeviations` documentation.

    This algorithm saves a rich set of data:

    - Individual histograms for each point along P1-P2 are saved in the given
      `directory`.
    - A Plotly figure of computed statistics is returned, including the
      deviations' mean, standard deviation, skewness and kurtosis.
    - The raw data is saved as object attributes (see below).

    The generated images (saved in `directory` with `height` x `width` pixels)
    can be stitched into a video using `pept.plots.make_video`.

    The extra keyword arguments ``**kwargs`` are passed to the histogram
    creation routine `pept.plots.histogram`. You can e.g. set the YAxis limits
    by adding `ylim = [0, 20]`.

    Attributes
    ----------
    deviations : list[(N,) np.ndarray]
        A list of deviations computed by `RelativeDeviations` at each point
        between P1 and P2.

    mean : (N,) np.ndarray
        A vector of mean tracer deviations at each point between P1 and P2.

    std : (N,) np.ndarray
        A vector of the tracer deviations' standard deviation at each point
        between P1 and P2.

    skew : (N,) np.ndarray
        A vector of the tracer deviations' adjusted skewness at each point
        between P1 and P2. A normal distribution has a value of 0; positive
        values indicate a longer right distribution tail; negative values
        indicate a heavier left tail.

    kurtosis : (N,) np.ndarray
        A vector of the tracer deviations' Fisher kurtosis at each point
        between P1 and P2. A normal distribution has a value of 0; positive
        values indicate a "thin" distribution; negative values indicate a
        heavy, wide distribution.

    Examples
    --------
    Consider a pipe-flow experiment, with tracers moving from side to side in
    multiple passes / streamlines. First locate the tracers, then split their
    trajectories into each individual pass:

    >>> import pept
    >>> from pept.tracking import *
    >>>
    >>> split_pipe = pept.Pipeline([
    >>>     Segregate(window = 10, max_distance = 20),  # Appends label column
    >>>     GroupBy("label"),                           # Splits into samples
    >>>     Reorient(),                                 # Align with X axis
    >>>     Center(),                                   # Center points at 0
    >>>     Stack(),
    >>> ])
    >>> streamlines = split_pipe.fit(trajectories)

    Now each sample in `streamlines` corresponds to a single tracer pass, e.g.
    `streamlines[0]` is the first pass, `streamlines[1]` is the second. The
    passes were reoriented and centred such that the pipe is aligned with the
    X axis.

    Now the `RelativeDeviationsLinear` reducer can be used to create images of
    the mixing between the pipe entrance and exit:

    >>> from pept.processing import RelativeDeviationsLinear
    >>> entrance = [-100, 0, 0]     # Pipe data was aligned with X and centred
    >>> exit = [100, 0, 0]
    >>> summary_fig = RelativeDeviationsLinear(
    >>>     directory = "deviations",   # Creates directory to save images
    >>>     p1 = entrance,
    >>>     p2 = exit,
    >>> ).fit(streamlines)
    >>> summary_fig.show()              # Summary statistics: mean, std, etc.

    Now the directory "deviations" was created inside your current working
    folder, and all relative deviation histograms were saved there as
    "frame0000.png", "frame0001.png", etc.
    You can stitch all images together into a video using
    `pept.plots.make_video`:

    >>> import pept
    >>> pept.plots.make_video(
    >>>     "deviations/frame*.png",
    >>>     output = "deviations/video.avi"
    >>> )

    The raw deviations and statistics can also be extracted directly:

    >>> mixing_algorithm = RelativeDeviationsLinear(
    >>>     directory = "deviations",   # Creates directory to save images
    >>>     p1 = entrance,
    >>>     p2 = exit,
    >>> )
    >>> fig = mixing_algorithm.fit(streamlines)
    >>> fig.show()

    >>> deviations = mixing_algorithm.deviations
    >>> mean = mixing_algorithm.mean
    >>> std = mixing_algorithm.std
    >>> skew = mixing_algorithm.skew
    >>> kurtosis = mixing_algorithm.kurtosis
    '''

    def __init__(
        self,
        directory,
        p1, p2,
        num_divisions = 50,
        max_distance = 10,
        height = 1000,
        width = 2000,
        prefix = "frame",
        **kwargs,
    ):
        self.directory = directory
        self.num_divisions = int(num_divisions)

        self.p1 = np.asarray(p1, dtype = float)
        self.p2 = np.asarray(p2, dtype = float)

        self.max_distance = float(max_distance)

        self.height = int(height)
        self.width = int(width)
        self.prefix = prefix
        self.kwargs = kwargs

        # Will be set in `fit`
        self.deviations = None
        self.mean = None
        self.std = None
        self.skew = None
        self.kurtosis = None


    def fit(self, trajectories, verbose = True):

        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        divisions = np.c_[
            np.linspace(self.p1[0], self.p2[0], self.num_divisions),
            np.linspace(self.p1[1], self.p2[1], self.num_divisions),
            np.linspace(self.p1[2], self.p2[2], self.num_divisions),
        ]

        axis = self.p2 - self.p1

        if verbose:
            divisions = tqdm(
                divisions,
                desc = "RelativeDeviationsLinear 1 / 3 :",
            )

        # Save vectors of deviations so that we can set the XAxis range
        self.deviations = []

        for i, div in enumerate(divisions):
            devs = RelativeDeviations(
                p1 = self.p1,
                p2 = div,
                ax1 = axis,
                ax2 = axis,
                max_distance = self.max_distance,
                histogram = False,
            ).fit(trajectories)

            self.deviations.append(devs)

        # Save histograms to disk
        if verbose:
            divisions = tqdm(
                divisions.iterable,
                desc = "RelativeDeviationsLinear 2 / 3 :",
            )

        xlim = [0, np.max([d.max() for d in self.deviations])]
        for i, div in enumerate(divisions):
            fig = plots.histogram(
                self.deviations[i],
                xlim = xlim,
                **self.kwargs,
            )

            travelled = np.linalg.norm(div - self.p1)
            fig.update_layout(
                title = f"Travelled {travelled:4.4f} mm",
                xaxis_title = "Deviation (mm)",
                yaxis_title = "Probability (%)",
            )

            fig.write_image(
                f"{self.directory}/{self.prefix}{i:0>4}.png",
                height = self.height,
                width = self.width,
            )

        # Compute summarised statistics about distributions
        if verbose:
            divisions = tqdm(
                divisions.iterable,
                desc = "RelativeDeviationsLinear 3 / 3 :",
            )

        self.mean = np.zeros(len(self.deviations))
        self.std = np.zeros(len(self.deviations))
        self.skew = np.zeros(len(self.deviations))
        self.kurtosis = np.zeros(len(self.deviations))

        for i, dev in enumerate(self.deviations):
            self.mean[i] = np.mean(dev)
            self.std[i] = np.std(dev)
            self.skew[i] = scipy.stats.skew(dev)
            self.kurtosis[i] = scipy.stats.kurtosis(dev)

        # Plot summarised statistics
        distance = np.linspace(
            0,
            np.linalg.norm(self.p2 - self.p1),
            self.num_divisions,
        )

        fig = make_subplots(rows = 2, cols = 1, subplot_titles = [
            "Mean Tracer Deviation Along P1-P2 Axis",
            "Distribution Skewness & Kurtosis (How Many Outliers?)",
        ])

        fig.add_trace(
            go.Scatter(
                x = distance,
                y = self.mean,
                mode = "lines",
                name = "Mean Deviation",
            ),
            row = 1, col = 1,
        )

        fig.add_trace(
            go.Scatter(
                x = distance,
                y = self.mean - self.std,
                mode = 'lines',
                marker_color = "#444",
                line_width = 0,
                showlegend = False,
            ),
            row = 1, col = 1,
        )

        fig.add_trace(
            go.Scatter(
                x = distance,
                y = self.mean + self.std,
                mode = 'lines',
                marker_color = "#444",
                line_width = 0,
                fillcolor = 'rgba(68, 68, 68, 0.3)',
                fill = 'tonexty',
                name = "Standard Deviation",
            ),
            row = 1, col = 1,
        )

        # Plot skewness and kurtosis
        fig.add_trace(
            go.Scatter(
                x = distance,
                y = self.skew,
                name = "Skewness",
            ),
            row = 2, col = 1,
        )

        fig.add_trace(
            go.Scatter(
                x = distance,
                y = self.kurtosis,
                name = "Kurtosis",
            ),
            row = 2, col = 1,
        )

        plots.format_fig(fig)
        fig.update_xaxes(title = "Length Along P1-P2 Axis (mm)")
        fig.update_layout(yaxis_title = "Deviation (mm)")

        return fig
