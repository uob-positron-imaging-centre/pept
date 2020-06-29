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


# File   : trajectory_separation.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 23.08.2019


import  os
import  time

import  numpy                       as      np
from    scipy.spatial               import  cKDTree
from    scipy.sparse                import  csr_matrix
from    scipy.sparse.csgraph        import  minimum_spanning_tree

from    tqdm                        import  tqdm

import  pept
import  hdbscan

from    .tco                        import  with_continuations
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
        dist, index = tree.query(pos, k = 1,  n_jobs = max_workers)

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




def segregate_trajectories(
    point_data,
    points_window,
    trajectory_cut_distance,
    min_trajectory_size = 5,
    as_list = False,
    return_mst = False
):
    '''Segregate the intertwined points from multiple trajectories into
    individual paths.

    The points in `point_data` (a numpy array or `pept.PointData`) are used to
    construct a minimum spanning tree in which every point can only be
    connected to `points_window` points around it - this "window" refers to the
    points in the initial data array, sorted based on the time column;
    therefore, only points within a certain timeframe can be connected. All
    edges (or "connections") in the minimum spanning tree that are larger than
    `trajectory_cut_distance` are removed (or "cut") and the remaining
    connected "clusters" are deemed individual trajectories if they contain
    more than `min_trajectory_size` points.

    The trajectory indices (or labels) are appended to `point_data`. That is,
    for each data point (i.e. row) in `point_data`, a label will be appended
    starting from 0 for the corresponding trajectory; a label of -1 represents
    noise. If `point_data` is a numpy array, a new numpy array is returned; if
    it is a `pept.PointData` instance, a new instance is returned.

    This function uses single linkage clustering with a custom metric for
    spatio-temporal data to segregate trajectory points. The single linkage
    clustering was optimised for this use-case: points are only connected if
    they are within a certain `points_window` in the time-sorted input array.
    Sparse matrices are also used for minimising the memory footprint.

    Parameters
    ----------
    point_data : (M, N>=4) numpy.ndarray or pept.PointData
        The points from multiple trajectories. Each row in `point_data` will
        have a timestamp and the 3 spatial coordinates, such that the data
        columns are [time, x_coord, y_coord, z_coord]. Note that `point_data`
        can have more data columns and they will simply be ignored.
    points_window : int
        Two points are "reachable" (i.e. they can be connected) if and only if
        they are within `points_window` in the time-sorted input `point_data`.
        As the points from different trajectories are intertwined (e.g. for two
        tracers A and B, the `point_data` array might have two entries for A,
        followed by three entries for B, then one entry for A, etc.), this
        should optimally be the largest number of points in the input array
        between two consecutive points on the same trajectory. If
        `points_window` is too small, all points in the dataset will be
        unreachable. Naturally, a larger `time_window` correponds to more pairs
        needing to be checked (and the function will take a longer to
        complete).
    trajectory_cut_distance : float
        Once all the closest points are connected (i.e. the minimum spanning
        tree is constructed), separate all trajectories that are further
        apart than `trajectory_cut_distance`.
    min_trajectory_size : float, default 5
        After the trajectories have been cut, declare all trajectories with
        fewer points than `min_trajectory_size` as noise.
    as_list : bool, default False
        If True, return a list of arrays, where each array contains the points
        in a single trajectory. In other words, return separate, single
        trajectories in a list. If False, return a single array of all points
        (if `point_data` was a `numpy.ndarray`) or a `pept.PointData`
        (if `point_data` was a `pept.PointData` instance).
    return_mst : bool, default False
        If `True`, the function will also return the minimum spanning tree
        constructed using the input `point_data`. This is a numpy array with
        columns [vertex1, vertex2, edge_length], where vertex1 and vertex2 are
        the indices in `point_data` of the connected points, and edge_length is
        the euclidian distance between them.

    Returns
    -------
    points_labelled: numpy.ndarray or pept.PointData or list of numpy.ndarray
        If `as_list` is `False`, this is the `point_data` array or
        `pept.PointData` instance with an extra column for the trajectory index
        (i.e. label) - the return type is similar to the input type. If
        `as_list` is `True`, this is a list of arrays, in which each array
        contains the points in a single_trajectory; these still include the
        trajectory label. A label value of `-1` indicates noise; the found
        trajectories are then labelled starting from 0.
    mst: numpy.ndarray, optional
        If `return_mst` is `True`, another numpy array is returned as a second
        variable containing the columns [vertex1, vertex2, edge_length], where
        vertex1 and vertex2 are the indices in `point_data` of the connected
        points, and edge_length is the euclidian distance between them.

    Raises
    ------
    ValueError
        If `point_data` is a numpy array with fewer than 4 columns.
    ValueError
        If `points_window` is smaller than 1.

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

    See Also
    --------
    connect_trajectories : Connect segregated trajectories based on tracer
                           signatures.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    '''

    # Check `point_data` is a numpy array or pept.PointData
    if isinstance(point_data, pept.PointData):
        pts = point_data.points
    else:
        pts = np.asarray(point_data)
        if pts.ndim != 2 or pts.shape[1] < 4:
            raise ValueError((
                "\n[ERROR]: `point_data` should have dimensions (M, N), where "
                f"N >= 4. Received {pts.shape}.\n"
            ))

    # Sort pts based on the time column (col 0) and create a C-ordered copy to
    # send to Cython.
    pts = np.asarray(pts[pts[:, 0].argsort()], dtype = float, order = "C")

    # Type-check the input parameters
    points_window = int(points_window)
    if points_window < 1:
        raise ValueError((
            "\n[ERROR]: `points_window` should be larger than 1! Received "
            f"{points_window}.\n"
        ))

    trajectory_cut_distance = float(trajectory_cut_distance)
    min_trajectory_size = int(min_trajectory_size)
    return_mst = bool(return_mst)

    # Calculate the sparse distance matrix between reachable points. This is an
    # optimised Cython function returning a sparse CSR matrix.
    distance_matrix = distance_matrix_reachable(pts, points_window)

    # Construct the minimum spanning tree from the sparse distance matrix. Note
    # that `mst` is also a sparse CSR matrix.
    mst = minimum_spanning_tree(distance_matrix)

    # Get the minimum spanning tree edges into the [vertex 1, vertex 2,
    # edge distance] format, then sort it based on the edge distance.
    mst = mst.tocoo()
    mst_edges = np.vstack((mst.row, mst.col, mst.data)).T
    mst_edges = mst_edges[mst_edges[:, 2].argsort()]

    # Create the single linkage tree from the minimum spanning tree edges using
    # internal hdbscan methods (because they're damn fast). This should be a
    # fairly quick step.
    single_linkage_tree = hdbscan._hdbscan_linkage.label(mst_edges)
    single_linkage_tree = hdbscan.plots.SingleLinkageTree(single_linkage_tree)

    # Cut the single linkage tree at `trajectory_cut_distance` and get the
    # cluster labels, setting clusters smaller than `min_trajectory_size` to
    # -1 (i.e. noise).
    labels = single_linkage_tree.get_clusters(
        trajectory_cut_distance,
        min_trajectory_size
    )

    # Append the labels to `pts`.
    pts = np.append(pts, labels[:, np.newaxis], axis = 1)

    # Returns based on as_list, return_mst and input data type
    if as_list:
        # Get a list of arrays for each trajectory
        separate_pts = pept.utilities.group_by_column(pts, -1)
        if return_mst:
            return separate_pts, mst_edges
        return separate_pts

    # If `point_data` was a `pept.PointData` instance, return a new
    # `pept.PointData` with the new label column.
    if isinstance(point_data, pept.PointData):
        point_data_labelled = pept.PointData(
            pts,
            sample_size = point_data.sample_size,
            overlap = point_data.overlap,
            verbose = False
        )
        if return_mst:
            return point_data_labelled, mst_edges
        else:
            return point_data_labelled
    elif return_mst:
        return pts, mst_edges
    return pts




def connect_trajectories(
    trajectories_points,
    max_time_difference,
    max_signature_difference,
    points_to_check = 50,
    signature_col = 4,
    label_col = -1,
    as_list = False
):
    '''Connect segregated trajectories based on tracer signatures.

    A pair of trajectories in `trajectories_points` will be connected if their
    ends have a timestamp difference that is smaller than `max_time_difference`
    and the difference between the signature averages of the closest
    `points_to_check` points is smaller than `max_signature_difference`.

    The `trajectories_points` are distinguished based on the trajectory
    indices in the data column `label_col`. This can be achieved using the
    `segregate_trajectories` function, which appends the labels to the data
    points.

    Because the tracer signature (e.g. cluster size in PEPT-ML) varies with the
    tracer position in the system, an average of `points_to_check` points
    is used for connecting pairs of trajectories.

    Parameters
    ----------
    trajectories_points : (M, N>=6) numpy.ndarray or pept.PointData
        A numpy array of points that have a timestamp, spatial coordinates,
        a tracer signature (such as cluster size in PEPT-ML) and a trajectory
        index (or label). The data columns in `trajectories_points` are then
        [time, x, y, z, ..., signature, ..., label, ...]. Note that the
        timestamps and spatial coordinates must be the first 4 columns, while
        the signature and label columns may be anywhere and are pointed at
        by `signature_col` and `label_col`.
    max_time_difference : float
        Only try to connect trajectories whose ends have a timestamp difference
        smaller than `max_time_difference`.
    max_signature_difference : float
        Connect two trajectories if the difference between the signature
        averages of the closest `points_to_check` is smaller than this.
    points_to_check : int, default 50
        The number of points used when computing the average tracer signature
        in one trajectory.
    signature_col : int, default 4
        The column in `trajectories_points` that contains the tracer
        signatures. The default is 4 (i.e. the signature comes right after
        the spatial coordinates).
    label_col : int, default -1
        The column in `trajectories_points` that contains the trajectory
        indices (labels). The default is -1 (i.e. the last column).
    as_list : bool, default False
        If True, return a list of arrays, where each array contains the points
        in a single trajectory. In other words, return separate, single
        trajectories in a list. If False, return a single array of all points
        (if `trajectories_points` was a `numpy.ndarray`) or a `pept.PointData`
        (if `trajectories_points` was a `pept.PointData` instance), but with
        labels changed to reflect the connected trajectories.

    Returns
    -------
    numpy.ndarray or pept.PointData or list of numpy.ndarray
        If `as_list` is True, return separate, single trajectories in a list.
        If `as_list` is False, return a single array of all points
        (if `trajectories_points` was a `numpy.ndarray`) or a `pept.PointData`
        (if `trajectories_points` was a `pept.PointData` instance), but with
        labels changed to reflect the connected trajectories.

    Raises
    ------
    ValueError
        If `point_data` is a numpy array with fewer than 6 columns.

    Notes
    -----
    The labels are changed in-place to reflect the connected trajectories. For
    example, if there are 3 trajectories with labels 0, 1, 2 and the first two
    are connected, then all points which previously had the label 1 will be
    changed to label 0; the last trajectory's label remains unchanged, 2.

    Examples
    --------
    [TODO] - add full tutorial page on Bham PIC GitHub page for this.

    See Also
    --------
    segregate_trajectories : Segregate the intertwined points from multiple
                             trajectories into individual paths.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    '''

    # Check `point_data` is a numpy array or pept.PointData
    if isinstance(trajectories_points, pept.PointData):
        trajs = trajectories_points.points
    else:
        trajs = np.asarray(trajectories_points, dtype = float, order = "C")
        if trajs.ndim != 2 or trajs.shape[1] < 6:
            raise ValueError((
                "\n[ERROR]: `trajectories_points` should have dimensions "
                f"(M, N), where N >= 6. Received {trajs.shape}.\n"
            ))

    # Type-check the input parameters
    max_time_difference = float(max_time_difference)
    max_signature_difference = float(max_signature_difference)
    points_to_check = int(points_to_check)
    signature_col = int(signature_col)
    label_col = int(label_col)
    as_list = bool(as_list)

    # Separate the trajs array into a list of individual trajectories based on
    # the `label_col`.
    trajectory_list = pept.utilities.group_by_column(trajs.copy(), label_col)

    trajectory_list = _connect_trajectories(
        trajectory_list,
        max_time_difference,
        max_signature_difference,
        points_to_check,
        signature_col,
        label_col
    )

    if as_list:
        return trajectory_list
    elif isinstance(trajectories_points, pept.PointData):
        trajectories_points_connected = pept.PointData(
            np.vstack(trajectory_list),
            sample_size = trajectories_points.sample_size,
            overlap = trajectories_points.overlap,
            verbose = False
        )
        return trajectories_points_connected
    else:
        return np.vstack(trajectory_list)




# Use tail-call optimisation from tco.py
@with_continuations()
def _connect_trajectories(
    trajectory_list,
    max_time_difference,
    max_signature_difference,
    points_to_check,
    signature_col = 4,
    label_col = -1,
    self = None
):
    number_of_trajectories = len(trajectory_list)

    # Check all pairs of trajectories. Each trajectory has two ends:
    # Traj1: [End11, End12]
    # Traj2: [End21, End22]
    for i in range(number_of_trajectories - 1):
        for j in range(i + 1, number_of_trajectories):

            t1 = trajectory_list[i]     # Traj1
            t2 = trajectory_list[j]     # Traj2

            # Try to connect End11 with End22
            # Check the time difference between End11 and End22 is small enough
            if time_difference(t1, t2) < max_time_difference:

                # Check the signature difference of the last `points_to_check`
                # points is small enough
                if signature_difference(t1, t2, points_to_check,
                    signature_col) < max_signature_difference:

                    # Assimilate the column labels
                    t1[:, label_col] = t2[0, label_col]

                    # Merge the two trajectories
                    connected_trajectory = np.append(t2, t1, axis = 0)

                    # Delete the individual trajectories and append the merged
                    # trajectories to `trajectory_list`
                    del trajectory_list[i], trajectory_list[j - 1]
                    trajectory_list.append(connected_trajectory)

                    # Call the function again (to reinitialise
                    # `number_of_trajectories`) using tail call optimisation
                    # from tco.py
                    return self(
                        trajectory_list,
                        max_time_difference,
                        max_signature_difference,
                        points_to_check,
                        signature_col,
                        label_col
                    )

            # Try to connect End12 with End21
            elif time_difference(t2, t1) < max_time_difference:

                if signature_difference(t2, t1, points_to_check,
                    signature_col) < max_signature_difference:

                    t2[:, label_col] = t1[0, label_col]
                    connected_trajectory = np.append(t1, t2, axis = 0)

                    del trajectory_list[i], trajectory_list[j - 1]
                    trajectory_list.append(connected_trajectory)

                    return self(
                        trajectory_list,
                        max_time_difference,
                        max_signature_difference,
                        points_to_check,
                        signature_col,
                        label_col
                    )

    return trajectory_list




def signature_difference(
    traj1,
    traj2,
    points_to_check,
    signature_col
):
    return np.abs(
        np.average(traj1[:points_to_check, signature_col]) - \
        np.average(traj2[-points_to_check:, signature_col])
    )




def time_difference(traj1, traj2):
    return np.abs(traj1[0, 0] - traj2[-1, 0])


