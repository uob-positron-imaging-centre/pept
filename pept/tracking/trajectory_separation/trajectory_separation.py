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
#    Copyright (C) 2019-2021 the pept developers
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


# File   : trajectory_separation.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 23.08.2019


import  os
import  warnings

import  numpy                       as      np
from    scipy.spatial               import  cKDTree
from    scipy.sparse.csgraph        import  minimum_spanning_tree

import  pept
import  hdbscan

from    .distance_matrix_reachable  import  distance_matrix_reachable




def trajectory_errors(
    true_positions,
    tracked_positions,
    averages = True,
    stds = False,
    errors = False,
    max_workers = None
):

    # true_positions, tracked_positions cols: [t, x, y, z, ...]
    true_positions = np.asarray(true_positions, dtype = float)
    tracked_positions = np.asarray(tracked_positions, dtype = float)

    # Construct k-d tree for fast nearest neighbour lookup.
    tree = cKDTree(true_positions[:, 1:4])

    # errors cols: [err, err_x, err_y, err_z]
    errors = np.zeros((len(tracked_positions), 4))

    if max_workers is None:
        max_workers = os.cpu_count()

    # Find closest true point to each tracked point.
    for i, pos in enumerate(tracked_positions[:, 1:4]):
        dist, index = tree.query(pos, k = 1, n_jobs = max_workers)

        errors[i, 0] = np.linalg.norm(pos - true_positions[index])
        errors[i, 1:4] = np.abs(pos - true_positions[index, 0:3])
        # errors[i, 1] = np.abs(pos[0] - true_positions[index, 0])
        # errors[i, 2] = np.abs(pos[1] - true_positions[index, 1])
        # errors[i, 3] = np.abs(pos[2] - true_positions[index, 2])

    # Append and return the results that were asked for.
    results = []

    if averages:
        results.append(np.average(errors, axis = 0))
    if stds:
        results.append(np.std(errors, axis = 0))
    if errors:
        results.append(errors)

    # If a single result was asked for, return it directly; otherwise, use a
    # tuple.
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)




class Segregate(pept.base.Reducer):
    '''Segregate the intertwined points from multiple trajectories into
    individual paths.

    Reducer signature:

    ::

              pept.PointData -> Segregate.fit -> pept.PointData
        list[pept.PointData] -> Segregate.fit -> pept.PointData
               numpy.ndarray -> Segregate.fit -> pept.PointData

    The points in `point_data` (a numpy array or `pept.PointData`) are used
    to construct a minimum spanning tree in which every point can only be
    connected to `points_window` points around it - this "window" refers to
    the points in the initial data array, sorted based on the time column;
    therefore, only points within a certain timeframe can be connected. All
    edges (or "connections") in the minimum spanning tree that are larger
    than `trajectory_cut_distance` are removed (or "cut") and the remaining
    connected "clusters" are deemed individual trajectories if they contain
    more than `min_trajectory_size` points.

    The trajectory indices (or labels) are appended to `point_data`. That
    is, for each data point (i.e. row) in `point_data`, a label will be
    appended starting from 0 for the corresponding trajectory; a label of
    -1 represents noise. If `point_data` is a numpy array, a new numpy
    array is returned; if it is a `pept.PointData` instance, a new instance
    is returned.

    This function uses single linkage clustering with a custom metric for
    spatio-temporal data to segregate trajectory points. The single linkage
    clustering was optimised for this use-case: points are only connected
    if they are within a certain `points_window` in the time-sorted input
    array. Sparse matrices are also used for minimising the memory
    footprint.

    Attributes
    ----------
    window : int
        Two points are "reachable" (i.e. they can be connected) if and only
        if they are within `points_window` in the time-sorted input
        `point_data`. As the points from different trajectories are
        intertwined (e.g. for two tracers A and B, the `point_data` array
        might have two entries for A, followed by three entries for B, then
        one entry for A, etc.), this should optimally be the largest number
        of points in the input array between two consecutive points on the
        same trajectory. If `points_window` is too small, all points in the
        dataset will be unreachable. Naturally, a larger `time_window`
        correponds to more pairs needing to be checked (and the function
        will take a longer to complete).

    cut_distance : float
        Once all the closest points are connected (i.e. the minimum
        spanning tree is constructed), separate all trajectories that are
        further apart than `trajectory_cut_distance`.

    min_trajectory_size : float, default 5
        After the trajectories have been cut, declare all trajectories with
        fewer points than `min_trajectory_size` as noise.

    See Also
    --------
    Reconnet : Connect segregated trajectories based on tracer signatures.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.

    Examples
    --------
    A typical workflow would involve transforming LoRs into points using some
    tracking algorithm. These points include all tracers moving through the
    system, being intertwined (e.g. for two tracers A and B, the `point_data`
    array might have two entries for A, followed by three entries for B, then
    one entry for A, etc.). They can be segregated based on position alone
    using this function; take for example two tracers that go downwards (below,
    'x' is the position, and in parens is the array index at which that point
    is found).

    ::

        `points`, numpy.ndarray, shape (10, 4), columns [time, x, y, z]:
            x (1)                       x (2)
             x (3)                     x (4)
               x (5)                 x (7)
               x (6)                x (9)
              x (8)                 x (10)

    >>> import pept.tracking.trajectory_separation as tsp
    >>> points_window = 10
    >>> trajectory_cut_distance = 15    # mm
    >>> segregated_trajectories = tsp.segregate_trajectories(
    >>>     points, points_window, trajectory_cut_distance
    >>> )

    ::

        `segregated_trajectories`, numpy.ndarray, shape (10, 5),
        columns [time, x, y, z, trajectory_label]:
            x (1, label = 0)            x (2, label = 1)
             x (3, label = 0)          x (4, label = 1)
               x (5, label = 0)      x (7, label = 1)
               x (6, label = 0)     x (9, label = 1)
              x (8, label = 0)      x (10, label = 1)
    '''

    def __init__(
        self,
        window,
        cut_distance,
        min_trajectory_size = 5,
    ):
        self.window = int(window)
        self.cut_distance = float(cut_distance)
        self.min_trajectory_size = int(min_trajectory_size)


    def fit(self, points):
        # Stack the input points into a single PointData
        if not isinstance(points, pept.PointData):
            points = pept.PointData(points)

        if len(points.points) == 0:
            return points.copy(
                data = points.points[0:0],
                columns = points.columns + ["label"],
            )

        pts = points.points

        # Sort pts based on the time column (col 0) and create a C-ordered copy
        # to send to Cython.
        pts = np.asarray(pts[pts[:, 0].argsort()], dtype = float, order = "C")

        # Calculate the sparse distance matrix between reachable points. This
        # is an optimised Cython function returning a sparse CSR matrix.
        distance_matrix = distance_matrix_reachable(pts, self.window)

        # Construct the minimum spanning tree from the sparse distance matrix.
        # Note that `mst` is also a sparse CSR matrix.
        mst = minimum_spanning_tree(distance_matrix)

        # Get the minimum spanning tree edges into the [vertex 1, vertex 2,
        # edge distance] format, then sort it based on the edge distance.
        mst = mst.tocoo()
        mst_edges = np.vstack((mst.row, mst.col, mst.data)).T
        mst_edges = mst_edges[mst_edges[:, 2].argsort()]

        # Ignore deprecation warning from HDBSCAN's use of `np.bool`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = DeprecationWarning)

            # Create the single linkage tree from the minimum spanning tree
            # edges using internal hdbscan methods (because they're damn fast).
            # This should be a fairly quick step.
            linkage_tree = hdbscan._hdbscan_linkage.label(mst_edges)
            linkage_tree = hdbscan.plots.SingleLinkageTree(linkage_tree)

            # Cut the single linkage tree at `trajectory_cut_distance` and get
            # the cluster labels, setting clusters smaller than
            # `min_trajectory_size` to -1 (i.e. noise).
            labels = linkage_tree.get_clusters(
                self.cut_distance,
                self.min_trajectory_size,
            )

        # Append the labels to `pts`.
        return points.copy(
            data = np.c_[pts, labels],
            columns = points.columns + ["label"],
        )




class Reconnect(pept.base.Reducer):
    '''Best-fit trajectory segment reconstruction based on time, distance and
    arbitrary tracer signatures.

    Reducer signature:

    ::

              pept.PointData -> Segregate.fit -> pept.PointData
        list[pept.PointData] -> Segregate.fit -> pept.PointData
               numpy.ndarray -> Segregate.fit -> pept.PointData

    After a trajectory segregation step (e.g. using ``Segregate``), you may be
    left with multiple smaller trajectory segments. Some trajectories can be
    reconstructed even when losing the tracers for a bit.

    When a tracer is lost for less than `tmax` time and `dmax` distance, its
    trajectory segments are reconnected; if multiple condidates are possible,
    the best fit is used.

    Multiple tracer signatures can be used to improve the reconnection step;
    supply them as data column names and difference thresholds, e.g. an extra
    keyword argument ``v = 1`` will join trajectories whose difference in
    velocity is smaller than 1 m/s.

    The last `num_points` points on a segment are averaged before they are
    connected with the first `num_points` on another segment.

    *New in pept-0.4.2*

    Examples
    --------
    Reconnect segments that are closer than 1 second in time and 0.1 m apart:

    >>> from pept.tracking import *
    >>> trajectories = Reconnect(tmax = 1000, dmax = 100).fit(segments)

    You can use the `cluster_size` (set by the ``Centroids`` filter) as a
    tracer signature; allow segments to be reconnected if the difference in
    their cluster size is < 100:

    >>> trajectories = Reconnect(1000, 100, cluster_size = 100).fit(segments)

    And a velocity `v` difference < 0.1:

    >>> Reconnect(1000, 100, cluster_size = 100, v = 0.1).fit(segments)

    '''

    def __init__(
        self,
        tmax,
        dmax,
        column = "label",
        num_points = 10,
        **signatures,
    ):
        self.column = column
        self.num_points = int(num_points)

        self.tmax = float(tmax)
        self.dmax = float(dmax)

        self.signatures = dict()
        for sig, thresh in signatures.items():
            self.signatures[sig] = float(thresh)


    def fit(self, points):
        points = pept.tracking.Stack().fit(points)
        if not isinstance(points, pept.PointData):
            points = pept.PointData(points)

        # Columns corresponding to the signatures
        sig_cols = [points.columns.index(sn) for sn in self.signatures.keys()]

        trajs = pept.tracking.SplitAll(self.column).fit(points)
        trajs.sort(key = lambda traj: traj["t"][0])

        # List of connections to do, list[tuple[int, int]]
        connections = []

        # Try to forward-connect the end of trajs[i] to the start of trajs[j]
        start_times = np.array([t["t"][0] for t in trajs])

        for i in range(len(trajs)):
            # Select all future trajectories whose start time is within tmax
            cur_traj = trajs[i]
            indices = np.argwhere(
                (start_times > cur_traj["t"][-1]) &
                (start_times - cur_traj["t"][-1] < self.tmax),
            ).flatten()

            # If no feasible times were found, carry on
            if not indices.any():
                continue

            # Compute connection costs between trajectory ends
            costs = []
            for j in indices:
                e2 = trajs[i].points[-self.num_points:].mean(axis = 0)
                e1 = trajs[j].points[:self.num_points].mean(axis = 0)

                # The first cost is the distance between traj ends; the rest
                # are the signature differences
                cost = [np.linalg.norm(e2[1:4] - e1[1:4])]
                for sc in sig_cols:
                    cost.append(np.abs(e2[sc] - e1[sc]))

                costs.append(cost)

            # Keep track of trajectory indices and associated costs
            costs = np.c_[indices, np.array(costs)]

            # Remove condidate connections that have costs larger than threshs
            selection = costs[:, 1] < self.dmax
            for i, sthresh in enumerate(self.signatures.values()):
                selection = selection & (costs[:, 2 + i] < sthresh)

            costs = costs[selection]

            # If no feasible connection was found, carry on
            if not len(costs):
                continue

            # Otherwise, establish connection with minimum overall cost
            best = costs[:, 1:].mean(axis = 1).argmin()
            connection_index = int(costs[best, 0])
            connections.append((i, connection_index))

        # Set connected labels
        if isinstance(self.column, str):
            label_col = points.columns.index(self.column)
        else:
            label_col = self.column

        for i1, i2 in connections:
            trajs[i2].points[:, label_col] = trajs[i1].points[0, label_col]

        # Stack trajectories and map labels from [0, 2, 2, 3, 0] to
        # [0, 1, 1, 2, 0]
        trajs = pept.tracking.Stack().fit(trajs)

        labels = trajs.points[:, label_col]
        _, ordered = np.unique(labels, return_inverse = True)
        trajs.points[:, label_col] = ordered

        return trajs
