#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_tracking.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 02.08.2021


'''Integration tests to ensure the `pept` base classes behave correctly and
offer consistent interfaces.
'''


import numpy as np
import pept

from pept.tracking import *


def test_birmingham_method():
    rng = np.random.default_rng(0)
    lines_raw = rng.random((5000, 7)) * 100
    lines = pept.LineData(lines_raw, sample_size = 200)

    location = BirminghamMethod(0.5, get_used = True).fit_sample(lines[0])
    print(location)

    locations = BirminghamMethod(0.5).fit(lines, "sequential")
    print(locations)


def test_cutpoints():
    rng = np.random.default_rng(0)
    lines_raw = rng.random((10, 7)) * 100
    lines = pept.LineData(lines_raw, sample_size=4)

    max_distance = 1000
    cutoffs = np.array([0, 100, 0, 100, 0, 100], dtype = float)
    cutpoints = pept.tracking.Cutpoints(max_distance, cutoffs)
    print(cutpoints)

    # Test `fit_sample`
    s1 = cutpoints.fit_sample(lines[0]).points
    s2 = pept.utilities.find_cutpoints(lines[0].lines, max_distance, cutoffs)
    assert (s1 == s2).all(), "Cutpoints not found correctly"

    # Test `fit`
    traversed = cutpoints.fit(lines)
    manual = [
        pept.utilities.find_cutpoints(ln.lines, max_distance, cutoffs)
        for ln in lines
    ]

    assert all([(t.points == m).all() for t, m in zip(traversed, manual)]), \
        "Traversed list of cutpoints not found correctly"


def test_minpoints():
    rng = np.random.default_rng(0)
    lines_raw = rng.random((10, 7)) * 100
    lines = pept.LineData(lines_raw, sample_size=4)

    max_distance = 1000
    cutoffs = np.array([0, 100, 0, 100, 0, 100], dtype = float)
    minpoints = pept.tracking.Minpoints(3, max_distance, cutoffs)
    print(minpoints)

    # Test `fit_sample`
    s1 = minpoints.fit_sample(lines[0]).points
    s2 = pept.utilities.find_minpoints(lines[0].lines, 3, max_distance,
                                       cutoffs)
    assert (s1 == s2).all(), "Cutpoints not found correctly"

    # Test `fit`
    traversed = minpoints.fit(lines)
    manual = [
        pept.utilities.find_minpoints(ln.lines, 3, max_distance, cutoffs)
        for ln in lines
    ]

    assert all([(t.points == m).all() for t, m in zip(traversed, manual)]), \
        "Traversed list of cutpoints not found correctly"


def test_hdbscan():
    rng = np.random.default_rng(0)
    lines_raw = rng.random((5000, 7)) * 100
    lines = pept.LineData(lines_raw, sample_size = 200)

    ex = "sequential"

    cutpoints = Cutpoints(0.5).fit(lines, ex)
    clustered = HDBSCAN(0.15, 2).fit(cutpoints, ex)
    print(clustered)

    clustered2 = HDBSCAN(0.15, 2).fit(clustered, ex)
    print(clustered2)


def test_centroids():
    rng = np.random.default_rng(0)
    points_raw = rng.random((10, 4)) * 100
    points = pept.PointData(points_raw, sample_size=4)

    f1 = pept.tracking.Centroids()
    print(f1)

    # Test `fit_sample`
    s1 = f1.fit_sample(points_raw).points
    s2 = points_raw.mean(axis = 0)
    assert (s1 == s2).all(), "Single sample geometric centroid"

    s1 = f1.fit_sample(points[0]).points
    s2 = points[0].points.mean(axis = 0)
    assert (s1 == s2).all(), "Single sample geometric centroid"

    # Test `fit`
    traversed = f1.fit(points)
    manual = [
        p.points.mean(axis = 0)
        for p in points
    ]

    assert all([(t.points == m).all() for t, m in zip(traversed, manual)]), \
        "Full `fit` traversal"

    # Test `fit_sample`
    s1 = f1.fit_sample(points[0]).points
    s2 = points_raw[:4].mean(axis = 0)
    assert (s2[:4] == s1[:, :4]).all(), "Single sample geometric centroid"

    # Test `fit`
    traversed = f1.fit(points)

    # Test different settings
    Centroids(error = True).fit_sample(points[0])
    Centroids(error = True, cluster_size = True).fit_sample(points[0])

    # Test weighted centroid computation
    points_raw = np.arange(50).reshape(10, 5)   # Last column is "weight"
    points_raw[:, -1] = 1.                      # Start with equal weights
    points = pept.PointData(points_raw,
                            columns = ["t", "x", "y", "z", "weight"],
                            sample_size = 4)

    # Test `fit_sample`
    s1 = f1.fit_sample(points_raw).points
    s2 = points_raw.mean(axis = 0)
    assert np.allclose(s1[:, :4], s2[:4]), "Single sample weighted centroid"

    s1 = f1.fit_sample(points[0]).points
    s2 = points[0].points.mean(axis = 0)
    assert np.allclose(s1[:, :4], s2[:4]), "Single sample weighted centroid"

    # Ensure "weight" is removed
    assert "weight" not in f1.fit_sample(points).columns

    # Test `fit`
    traversed = f1.fit(points)

    # Test different settings
    Centroids(error = True).fit_sample(points[0])
    Centroids(error = True, cluster_size = True).fit_sample(points[0])


def test_stack():
    rng = np.random.default_rng(0)
    points_raw = rng.random((10, 4)) * 100
    lines_raw = rng.random((10, 7)) * 500

    points = pept.PointData(points_raw, sample_size=4)
    lines = pept.LineData(lines_raw, sample_size=4)

    # Test it returns points back
    p = Stack().fit(points)
    assert p is points, "Stack did not return a single PointData back"

    # Test it returns lines back
    ls = Stack().fit(lines)
    assert ls is lines, "Stack did not return a single LineData back"

    # Test it concatenates a list of two points
    points2 = Stack().fit([points, points])
    assert np.all(points2.points[:10] == points.points[:10])

    # Test it concatenates a list of two lines
    lines2 = Stack().fit([lines, lines])
    assert np.all(lines2.lines[:10] == lines.lines[:10])

    # Test list[list] flattening
    assert Stack().fit([[1, 2, 3]]) == [1, 2, 3], "List flattening wrong"


def test_split_labels():
    rng = np.random.default_rng(0)
    points_raw = rng.random((10, 4)) * 100
    labels = rng.integers(3, size=10)
    line_index = rng.integers(10, size=10)

    points = pept.PointData(
        np.c_[points_raw, labels, line_index],
        columns = ["t", "x", "y", "z", "label", "line_index"],
    )
    points.samples_indices = [[0, 10], [5, 5], [5, 10]]

    # Check each split label
    split = SplitLabels().fit_sample(points[0])
    assert np.all(split[0].points[:, :4] == points_raw[labels == 0])
    assert np.all(split[1].points[:, :4] == points_raw[labels == 1])
    assert np.all(split[2].points[:, :4] == points_raw[labels == 2])

    # Check with empty sample
    empty_split = SplitLabels().fit_sample(points[1])
    assert len(empty_split[0].data) == 0

    # Extracting `_lines`
    lines_raw = rng.random((10, 7)) * 500
    lines = pept.LineData(lines_raw, sample_size=4)
    points.attrs["_lines"] = lines

    splines = SplitLabels().fit_sample(points[0])
    assert "_lines" in splines[0].attrs

    splines = SplitLabels(extract_lines = True).fit_sample(points[0])
    assert isinstance(splines[0], pept.LineData)

    # Test different settings
    SplitLabels().fit(points, "sequential")
    SplitLabels(remove_labels = False).fit(points, "sequential")
    SplitLabels(noise = True).fit(points, "sequential")
    SplitLabels(extract_lines = True).fit(points, "sequential")


def test_split_all():
    rng = np.random.default_rng(0)
    points_raw = rng.random((10, 4)) * 100
    labels = rng.integers(3, size=10)
    line_index = rng.integers(10, size=10)

    points = pept.PointData(
        np.c_[points_raw, labels, line_index],
        columns = ["t", "x", "y", "z", "label", "line_index"],
    )
    points.samples_indices = [[0, 10], [5, 5], [5, 10]]

    # Check each split label
    split = SplitAll("label").fit(points)
    assert np.all(split[0].points[:, :4] == points_raw[labels == 0])
    assert np.all(split[1].points[:, :4] == points_raw[labels == 1])
    assert np.all(split[2].points[:, :4] == points_raw[labels == 2])

    # Check with empty sample
    empty_split = SplitLabels().fit_sample(points[1])
    assert len(empty_split[0].data) == 0

    # Check using numeric index
    split_str = SplitAll("label").fit(points)
    split_idx = SplitAll(4).fit(points)

    assert np.all(split_str[0].points == split_idx[0].points)
    assert np.all(split_str[1].points == split_idx[1].points)
    assert np.all(split_str[2].points == split_idx[2].points)

    # Testing different settings
    SplitAll("label").fit([points])
    SplitAll("label").fit([points, points])
    SplitAll(4).fit(points.points)


def test_peptml():
    rng = np.random.default_rng(0)
    lines_raw = rng.random((5000, 7)) * 100
    lines = pept.LineData(lines_raw, sample_size = 200)

    ex = "sequential"

    cutpoints = Cutpoints(0.5).fit(lines, ex)
    clustered = HDBSCAN(0.15, 2).fit(cutpoints, ex)
    centres = (SplitLabels() + Centroids() + Stack(30, 29)).fit(clustered, ex)
    clustered2 = HDBSCAN(0.6, 2).fit(centres, ex)
    centres2 = (SplitLabels() + Centroids()).fit(clustered2, ex)
    print(centres2)


def test_lines_centroids():
    rng = np.random.default_rng(0)
    lines_raw = rng.random((1000, 7)) * 100
    lines = pept.LineData(lines_raw, sample_size = 200)

    LinesCentroids().fit_sample(lines)

    ex = "sequential"
    LinesCentroids().fit(lines, ex)
    LinesCentroids().fit(lines[0:0], ex)


def test_condition():
    rng = np.random.default_rng(0)
    points_raw = rng.random((10, 4)) * 100
    labels = rng.integers(3, size=10)

    points = pept.PointData(
        np.c_[points_raw, labels],
        columns = ["t", "x", "y", "z", "label"],
    )
    points.samples_indices = [[0, 10], [5, 5], [5, 10]]

    cp = Condition("x < 50").fit_sample(points)
    assert np.all(cp.data == points.data[
        points.points[:, points.columns.index("x")] < 50
    ])

    cp2 = Condition("'2' < 50").fit_sample(points)
    cp3 = Condition("50 > '2'").fit_sample(points)
    assert np.allclose(cp2.data, points.data[points.data[:, 2] < 50])
    assert np.allclose(cp2.data, cp3.data)

    # Testing different settings
    Condition("np.isfinite('x')").fit(points)
    Condition("'x' < 'y'").fit(points)
    Condition("x < 2, 'x' > 0, 1 > 'x'").fit(points)
    Condition(lambda arr: arr[:, 0] > 10).fit(points)
    Condition(lambda x: x[:, -1] < 50, 'x > 10').fit(points)


def test_swap():
    rng = np.random.default_rng(0)
    points_raw = rng.random((10, 4)) * 100
    labels = rng.integers(3, size=10)

    points = pept.PointData(
        np.c_[points_raw, labels],
        columns = ["t", "x", "y", "z", "label"],
    )
    points.samples_indices = [[0, 10], [5, 5], [5, 10]]

    # Simple, single swap
    p2 = Swap("y, z").fit_sample(points.copy())
    assert np.all(p2["y"] == points["z"]), "Swap not done"
    assert np.all(p2["z"] == points["y"]), "Swap not done"

    # Single swap with quoted column names
    p2 = Swap("'y', 'z'").fit_sample(points.copy())
    assert np.all(p2["y"] == points["z"]), "Swap not done"
    assert np.all(p2["z"] == points["y"]), "Swap not done"

    # Single swap with quoted column indices
    p2 = Swap("'2', '3'").fit_sample(points.copy())
    assert np.all(p2["y"] == points["z"]), "Swap not done"
    assert np.all(p2["z"] == points["y"]), "Swap not done"

    # Testing different settings
    Swap("y, z").fit(points)
    Swap("label, 'z'").fit(points)
    Swap("'0', '1'", "'y', 'z'", "x, z").fit(points)


def test_remove():
    rng = np.random.default_rng(0)
    points_raw = rng.random((10, 4)) * 100
    labels = rng.integers(3, size=10)

    points = pept.PointData(
        np.c_[points_raw, labels, labels],
        columns = ["t", "x", "y", "z", "label", "label2"],
    )
    points.samples_indices = [[0, 10], [5, 5], [5, 10]]

    rm = Remove("label").fit_sample(points)
    assert "label" not in rm.columns
    assert rm.points.shape[1] == 5

    rm = Remove("label*").fit_sample(points)
    assert "label" not in rm.columns
    assert "label2" not in rm.columns
    assert rm.points.shape[1] == 4

    # Testing different settings
    Remove(0).fit(points, "sequential")
    Remove(-1).fit(points, "sequential")
    Remove("label", "label2").fit(points, "sequential")
    Remove(0, "label").fit(points, "sequential")


def test_voxelizer():
    rng = np.random.default_rng(0)
    lines_raw = rng.random((1000, 7)) * 100
    lines = pept.LineData(lines_raw, sample_size = 200)

    vox = Voxelize((20, 20, 20)).fit_sample(lines)
    assert "_lines" in vox.attrs

    ex = "sequential"
    LinesCentroids().fit(lines, ex)
    LinesCentroids().fit(lines[0:0], ex)


def test_interpolate():
    points_raw = np.arange(60).reshape(10, 6)

    points = pept.PointData(
        points_raw,
        columns = ["t", "x", "y", "z", "label", "line_index"],
    )
    points.samples_indices = [[0, 10], [5, 5], [5, 10]]

    # Interpolate at double sampling rate
    half_interpolator = Interpolate((points_raw[1, 0] - points_raw[0, 0]) / 2)
    interp = half_interpolator.fit_sample(points)

    assert interp.points[1, 2] == (points_raw[0, 2] + points_raw[1, 2]) / 2

    # Testing different settings
    Interpolate(3., kind = "cubic").fit(points, "sequential")
    Interpolate(10., kind = "nearest").fit(points, "sequential")


def test_velocity():
    rng = np.random.default_rng(0)
    points_raw = rng.random((10, 4)) * 100
    points = pept.PointData(points_raw, sample_size=4)

    vs = Velocity(5).fit_sample(points)
    assert "vx" in vs.columns
    assert "vy" in vs.columns
    assert "vz" in vs.columns

    assert "v" in Velocity(5, absolute = True).fit_sample(points).columns

    # Testing different settings
    Velocity(3).fit(points, "sequential")
    Velocity(window = 9, degree = 5).fit(points, "sequential")


def test_segregate():
    rng = np.random.default_rng(0)
    points_raw = rng.random((100, 4)) * 100
    points = pept.PointData(points_raw, sample_size=4)

    se = Segregate(20, cut_distance = np.inf).fit(points)
    assert np.allclose(se.points[:, -1], 0.)

    # Generate trajectory formed of two sections, apart in space and time
    rng = np.random.default_rng(0)
    section1 = np.c_[
        np.arange(100),
        np.sin(np.linspace(-10, 10, 100)),
        np.sin(np.linspace(-10, 10, 100)),
        np.sin(np.linspace(-10, 10, 100)),
    ]

    section2 = np.c_[
        np.arange(1000, 1100),
        5 + np.sin(np.linspace(-10, 10, 100)),
        5 + np.sin(np.linspace(-10, 10, 100)),
        5 + np.sin(np.linspace(-10, 10, 100)),
    ]

    trajectories = pept.PointData(np.vstack((section1, section2)))
    se = Segregate(20, cut_distance = 1).fit(trajectories)
    assert len(np.unique(se.points[:, -1])) == 2

    # Generate trajectory formed of two sections, apart only in time
    rng = np.random.default_rng(0)
    section1 = np.c_[
        np.arange(100),
        np.sin(np.linspace(-10, 10, 100)),
        np.sin(np.linspace(-10, 10, 100)),
        np.sin(np.linspace(-10, 10, 100)),
    ]

    section2 = np.c_[
        np.arange(1000, 1100),
        np.sin(np.linspace(10, 20, 100)),
        np.sin(np.linspace(10, 20, 100)),
        np.sin(np.linspace(10, 20, 100)),
    ]

    trajectories = pept.PointData(np.vstack((section1, section2)))

    # Segregate without time -> single trajectory
    se = Segregate(
        20,
        cut_distance = 1,
    ).fit(trajectories)
    assert len(np.unique(se.points[:, -1])) == 1

    # Segregate with time -> correct, two trajectories
    se = Segregate(
        20,
        cut_distance = 1,
        max_time_interval = 10,
    ).fit(trajectories)
    assert len(np.unique(se.points[:, -1])) == 2

    # Testing different settings
    Segregate(5, 10, 15, 20).fit(points)
    Segregate(5, 10, 15).fit(points)
    Segregate(1, 1).fit(points)


def test_fpi():
    rng = np.random.default_rng(0)
    lines_raw = rng.random((1000, 7)) * 100
    lines = pept.LineData(lines_raw, sample_size = 200)

    ex = "sequential"
    voxels = Voxelize((50, 50, 50)).fit(lines, ex)
    positions = FPI().fit(voxels, ex)
    print(positions)


def test_reorient():
    rng = np.random.default_rng(0)

    # Generate points spread out differently in each dimension
    points_raw = [1, 100, 200, 300] * rng.random((1000, 4))

    spreads = points_raw.std(axis = 0)[1:]
    assert np.all(spreads.argsort() == [0, 1, 2])

    # Reorient points so that most spread out dimension becomes X
    points = pept.PointData(points_raw)

    reo = Reorient("xyz").fit(points)
    spreads = reo.points.std(axis = 0)[1:]
    assert np.all(spreads.argsort() == [2, 1, 0])

    reo = Reorient("yzx").fit(points)
    spreads = reo.points.std(axis = 0)[1:]
    assert np.all(spreads.argsort() == [0, 2, 1])

    # Testing different settings
    Reorient("zyx").fit(points_raw)
    Reorient(
        basis = reo.attrs["basis"],
        origin = reo.attrs["origin"],
    ).fit(points_raw)


def test_remove_static():
    # Generate trajectory that is moving, then almost static, then moving
    rng = np.random.default_rng(0)
    trajectories = np.vstack((
        np.c_[np.arange(0, 1000), rng.uniform(-50, 50, (1000, 3))],
        np.c_[np.arange(1000, 2000), rng.uniform(-5, 5, (1000, 3))],
        np.c_[np.arange(2000, 3000), rng.uniform(-50, 50, (1000, 3))],
        np.c_[np.arange(3000, 4000), rng.uniform(-5, 5, (1000, 3))],
    ))
    trajectories = pept.PointData(trajectories)

    # Good use
    kept_trajectories = RemoveStatic(
        time_window = 200,
        max_distance = 20,
    ).fit(trajectories)
    assert len(kept_trajectories.points) > 0

    # Time window larger than entire dataset
    kept_trajectories = RemoveStatic(
        time_window = 20_000,
        max_distance = 20,
    ).fit(trajectories)
    assert len(kept_trajectories.points) > 0
