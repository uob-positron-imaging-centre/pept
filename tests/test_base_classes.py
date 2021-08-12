#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_base_classes.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 02.08.2021


'''Integration tests to ensure the `pept` base classes behave correctly and
offer consistent interfaces.
'''


import pytest

import numpy as np
import pept


def test_point_data():
    # Test simple sample size, no overlap
    points_raw = np.arange(40).reshape(10, 4)
    points = pept.PointData(points_raw, sample_size=4)
    print(points)

    assert (points[0].points == points_raw[:4]).all(), "Incorrect 1st sample"
    assert (points[1].points == points_raw[4:8]).all(), "Incorrect 2nd sample"
    assert len(points) == 2, "Incorrect number of samples"

    # Test changing sample size and overlap (int)
    points.sample_size = 3
    points.overlap = 2
    assert (points[0].points == points_raw[:3]).all()
    assert (points[1].points == points_raw[1:4]).all()
    assert len(points) == 8, "Incorrect number of samples after overlap"

    # Test changing sample size to List[Int]
    points.sample_size = [3, 4, 2, 0]
    assert points.overlap is None, "Overlap was not set to None"
    assert len(points) == 4, "Incorrect number of samples"
    assert (points[0].points == points_raw[:3]).all(), "List sample size"
    assert (points[1].points == points_raw[3:7]).all(), "List sample size"
    assert (points[2].points == points_raw[7:9]).all(), "List sample size"
    assert (points[3].points == points_raw[9:9]).all(), "List sample size"

    # Test copying
    assert points.copy() is not points, "Copy is not deep"
    assert (points.copy().points == points.points).all(), "Incorrect copying"

    # Test illegal changes to sample size and overlap
    with pytest.raises(ValueError):
        points.sample_size = 3
        points.overlap = 3

    with pytest.raises(ValueError):
        points.sample_size = 0
        points.overlap = 3
        points.sample_size = 3

    with pytest.raises(ValueError):
        points.sample_size = -1

    # Test illegal array shapes
    with pytest.raises(ValueError):
        pept.PointData(np.arange(12))

    with pytest.raises(ValueError):
        pept.PointData(np.arange(12).reshape(4, 3))

    with pytest.raises(ValueError):
        pept.PointData(np.arange(12).reshape(2, 2, 3))


def test_line_data():
    # Test simple sample size, no overlap
    lines_raw = np.arange(70).reshape(10, 7)
    lines = pept.LineData(lines_raw, sample_size=4)
    print(lines)

    assert (lines[0].lines == lines_raw[:4]).all(), "Incorrect first sample"
    assert (lines[1].lines == lines_raw[4:8]).all(), "Incorrect second sample"
    assert len(lines) == 2, "Incorrent number of samples"

    # Test copying
    assert lines.copy() is not lines, "Copy is not deep"
    assert (lines.copy().lines == lines.lines).all(), "Incorrect copying"

    # Test changing sample size and overlap (int)
    lines.sample_size = 3
    lines.overlap = 2
    assert (lines[0].lines == lines_raw[:3]).all(), "Incorrect ssize changing"
    assert (lines[1].lines == lines_raw[1:4]).all(), "Incorrect overlapping"
    assert len(lines) == 8, "Incorrect number of samples after overlap"

    # Test changing sample size to List[Int]
    lines.sample_size = [3, 4, 2, 0]
    assert lines.overlap is None, "Overlap was not set to None"
    assert len(lines) == 4, "Incorrect number of samples"
    assert (lines[0].lines == lines_raw[:3]).all(), "List sample size"
    assert (lines[1].lines == lines_raw[3:7]).all(), "List sample size"
    assert (lines[2].lines == lines_raw[7:9]).all(), "List sample size"
    assert (lines[3].lines == lines_raw[9:9]).all(), "List sample size"

    # Test illegal changes to sample size and overlap
    with pytest.raises(ValueError):
        lines.sample_size = 3
        lines.overlap = 3

    with pytest.raises(ValueError):
        lines.sample_size = 0
        lines.overlap = 3
        lines.sample_size = 3

    with pytest.raises(ValueError):
        lines.sample_size = -1

    # Test illegal array shapes
    with pytest.raises(ValueError):
        pept.LineData(np.arange(12))

    with pytest.raises(ValueError):
        pept.LineData(np.arange(12).reshape(2, 6))

    with pytest.raises(ValueError):
        pept.LineData(np.arange(12).reshape(2, 2, 3))


def test_voxels():
    voxels_raw = np.arange(125).reshape(5, 5, 5)
    xlim = [10, 20]
    ylim = [-10, 0]
    zlim = [20, 30]

    voxels = pept.Voxels(voxels_raw, xlim, ylim, zlim)
    print(voxels)

    assert float(voxels.sum()) == float(voxels_raw.sum())


def test_voxel_data():
    lines_raw = np.arange(70).reshape(10, 7)
    lines = pept.LineData(lines_raw, sample_size=4)

    resolution = (5, 5, 5)
    xlim = [0, 100]
    ylim = [0, 100]
    zlim = [0, 100]
    voxel_data = pept.VoxelData(lines, resolution, xlim, ylim, zlim)
    print(voxel_data)

    # Test copying
    assert voxel_data.copy() is not voxel_data, "Copy is not deep"
    assert (voxel_data.copy()[0] == voxel_data[0]).all(), "Incorrent copying"

    # Test a single sample voxellisation is done correctly
    assert (voxel_data[0] == pept.Voxels.from_lines(
        lines[0], resolution, xlim, ylim, zlim)
    ).all()

    # Test traversal is done correctly
    traversed = voxel_data.traverse()
    manual = [
        pept.Voxels.from_lines(ln, resolution, xlim, ylim, zlim)
        for ln in lines
    ]

    assert all([(t == m).all() for t, m in zip(traversed, manual)]), \
        "Traversed list of voxels not found correctly"


class SomeSamples(pept.base.IterableSamples):
    def __init__(self, arr, sample_size, overlap):
        self._data_samples = arr
        pept.base.IterableSamples.__init__(self, arr, sample_size, overlap)


def test_iterable_samples_subclass():

    # Test simple sample size, no overlap
    array_raw = np.arange(20).reshape(10, 2)
    samples = SomeSamples(array_raw, sample_size=4, overlap=0)
    assert (samples[0].data == array_raw[:4]).all(), "Incorrect first sample"
    assert (samples[1].data == array_raw[4:8]).all(), "Incorrect second sample"
    assert len(samples) == 2, "Incorrent number of samples"

    # Test changing sample size and overlap (int)
    samples.sample_size = 3
    samples.overlap = 2
    assert (samples[0].data == array_raw[:3]).all(), "Incorrect sample size"
    assert (samples[1].data == array_raw[1:4]).all(), "Incorrect overlapping"
    assert len(samples) == 8, "Incorrect number of samples after overlap"

    # Test changing sample size to List[Int]
    samples.sample_size = [3, 4, 2, 0]
    assert samples.overlap is None, "Overlap was not set to None"
    assert len(samples) == 4, "Incorrect number of samples"
    assert (samples[0].data == array_raw[:3]).all(), "Incorrect sample size"
    assert (samples[1].data == array_raw[3:7]).all()
    assert (samples[2].data == array_raw[7:9]).all()
    assert (samples[3].data == array_raw[9:9]).all()

    # Test illegal changes to sample size and overlap
    with pytest.raises(ValueError):
        samples.sample_size = 3
        samples.overlap = 3

    with pytest.raises(ValueError):
        samples.sample_size = 0
        samples.overlap = 3
        samples.sample_size = 3

    with pytest.raises(ValueError):
        samples.sample_size = -1


class SomeAsyncSamples(pept.base.AsyncIterableSamples):
    def __init__(self, samples, function):
        pept.base.AsyncIterableSamples.__init__(self, samples, function)


def f(x):
    return 2 * x.data


def test_async_iterable_samples_subclass():

    # Test simple sample size, no overlap
    array_raw = np.arange(20).reshape(10, 2)
    samples_raw = SomeSamples(array_raw, sample_size=4, overlap=0)
    samples = SomeAsyncSamples(samples_raw, f)
    assert (samples[0] == f(samples_raw[0])).all(), "Incorrect 1st sample"
    assert (samples[1] == f(samples_raw[1])).all(), "Incorrect 2nd sample"
    assert len(samples) == 2, "Incorrent number of samples"

    # Test changing sample size and overlap of the unprocessed samples
    samples_raw.sample_size = 3
    samples_raw.overlap = 2
    assert (samples[0] == f(samples_raw[0])).all(), "Sample sizing"
    assert (samples[1] == f(samples_raw[1])).all(), "Overlapping"
    assert len(samples) == 8, "Incorrect number of samples after overlap"


def test_pipeline():

    class F1(pept.base.LineDataFilter):
        def fit_sample(self, sample_lines):
            sample_lines.lines[:] += 1
            sample_lines.attr1 = "New attribute added by F1"
            return sample_lines


    class F2(pept.base.LineDataFilter):
        def fit_sample(self, sample_lines):
            sample_lines.lines[:] += 2
            sample_lines.attr2 = "New attribute added by F2"
            return sample_lines


    class R1(pept.base.Reducer):
        def fit(self, lines):
            return tuple(lines)


    class R2(pept.base.Reducer):
        def fit(self, lines):
            return pept.LineData(lines)


    # Generate some dummy LineData
    lines_raw = np.arange(70).reshape(10, 7)
    lines = pept.LineData(lines_raw, sample_size=4)

    # Test pipeline creation
    assert isinstance(F1() + F2(), pept.base.Pipeline)
    assert isinstance(pept.base.Pipeline([F1(), F2()]), pept.base.Pipeline)
    assert isinstance(F1() + F2() + R1(), pept.base.Pipeline)

    # Test fit_sample
    pipe = F1() + F2()
    print(pipe)

    lp1 = pipe.fit_sample(lines[0]).lines
    lp2 = F2().fit_sample(F1().fit_sample(lines[0])).lines
    assert (lp1 == lp2).all(), "Apply simple pipeline steps manually"

    pipe = F1() + F2() + R1()
    print(pipe)

    lp1 = pipe.fit_sample(lines[0])
    lp2 = F1().fit_sample(lines[0])
    lp2 = F2().fit_sample(lp2)
    lp2 = R1().fit([lp2])

    assert isinstance(lp1, tuple), "Final pipeline reducer to tuple"
    assert isinstance(lp2, tuple), "Final manual reducer to tuple"
    assert (lp1[0].lines == lp2[0].lines).all(), "Apply steps manually"

    # Test the attribute is added by the first filter
    assert hasattr(F1().fit_sample(lines[0]), "attr1")
    assert hasattr(pept.base.Pipeline([F1()]).fit_sample(lines[0]), "attr1")

    # Test fit
    # Simple filter-only pipeline
    pipe = F1() + F2()

    lp1 = pipe.fit(lines)
    lp2 = F2().fit(F1().fit(lines))
    assert isinstance(lp1, list)
    assert isinstance(lp2, list)
    assert len(lp1) == len(lp2) == len(lines)

    assert all([(l1.lines == l2.lines).all() for l1, l2 in zip(lp1, lp2)])

    # Pipeline ending in reducer
    pipe = F1() + F2() + R1()
    print(pipe)

    lp1 = pipe.fit(lines)
    lp2 = F1().fit(lines)
    lp2 = F2().fit(lp2)
    lp2 = R1().fit(lp2)

    assert isinstance(lp1, tuple)
    assert isinstance(lp2, tuple)
    assert len(lp1) == len(lp2) == len(lines)

    assert all([(l1.lines == l2.lines).all() for l1, l2 in zip(lp1, lp2)])

    # Complex pipeline
    pipe = F1() + F2() + R2() + F1() + R1()
    print(pipe)

    lp1 = pipe.fit(lines)
    lp2 = F1().fit(lines)
    lp2 = F2().fit(lp2)
    lp2 = R2().fit(lp2)
    lp2 = F1().fit(lp2)
    lp2 = R1().fit(lp2)

    assert isinstance(lp1, tuple)
    assert isinstance(lp2, tuple)
    assert len(lp1) == len(lp2) == len(lines)

    assert all([(l1.lines == l2.lines).all() for l1, l2 in zip(lp1, lp2)])

    # Test the attribute is added by the first filter
    assert hasattr(F1().fit(lines)[0], "attr1")
    assert hasattr(pept.base.Pipeline([F1()]).fit(lines)[0], "attr1")
