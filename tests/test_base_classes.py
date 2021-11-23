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
    assert np.all(points["t"] == points_raw[:, 0]), "Incorrect string indexing"

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
    assert points.copy().points is not points.points, "Copy is not deep"
    assert points.copy(deep=False).points is points.points, "Not shallow copy"
    assert (points.copy().points == points.points).all(), "Incorrect copying"

    assert np.all(points.copy().samples_indices == points.samples_indices)
    points.samples_indices = [[0, 5], [5, 5], [5, 10]]
    assert np.all(points.copy().samples_indices == points.samples_indices)

    # Test different constructors: copy, iterable, numpy-like
    points_raw = np.arange(50).reshape(10, 5)
    columns = ["t", "x", "y", "z", "error"]
    points = pept.PointData(points_raw, columns = columns)

    pept.PointData(points)
    pept.PointData([points, points])
    pept.PointData([[1, 2, 3, 4], [1, 2, 3, 4]])

    # Test unnamed columns
    pept.PointData([range(5), range(5)])
    pept.PointData([range(5)], columns = ["a", "b", "c", "d", "e"])

    # Test columns propagation
    assert "error" in pept.PointData(points).columns
    assert "error" in pept.PointData([points, points]).columns

    # Test attrs propagation
    points.attrs["_lines"] = 123
    points.attrs["_attr2"] = [1, 2, 3]

    assert "_lines" in pept.PointData(points).attrs
    assert "_attr2" in pept.PointData([points, points]).attrs
    assert "_lines" in points[0].attrs
    assert "_attr2" in points.copy().attrs

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
    assert np.all(lines["t"] == lines_raw[:, 0]), "Incorrect string indexing"

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

    # Test copying
    assert lines.copy().lines is not lines.lines, "Copy is not deep"
    assert lines.copy(deep=False).lines is lines.lines, "Not shallow copy"
    assert (lines.copy().lines == lines.lines).all(), "Incorrect copying"

    assert np.all(lines.copy().samples_indices == lines.samples_indices)
    lines.samples_indices = [[0, 5], [5, 5], [5, 10]]
    assert np.all(lines.copy().samples_indices == lines.samples_indices)

    # Test different constructors: copy, iterable, numpy-like
    lines_raw = np.arange(80).reshape(10, 8)
    columns = ["t", "x1", "y1", "z1", "x2", "y2", "z2", "error"]
    lines = pept.LineData(lines_raw, columns = columns)

    pept.LineData(lines)
    pept.LineData([lines, lines])
    pept.LineData([range(7), range(7)])

    # Test unnamed columns
    pept.LineData([range(8), range(8)])
    pept.LineData([range(7)], columns = ["a", "b", "c", "d", "e", "f", "g",
                                         "h", "i"])

    # Test columns propagation
    assert "error" in pept.LineData(lines).columns
    assert "error" in pept.LineData([lines, lines]).columns

    # Test attrs propagation
    lines.attrs["_lines"] = 123
    lines.attrs["_attr2"] = [1, 2, 3]

    assert "_lines" in pept.LineData(lines).attrs
    assert "_attr2" in pept.LineData([lines, lines]).attrs
    assert "_lines" in lines[0].attrs
    assert "_attr2" in lines.copy().attrs

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


def f(x):
    return 2 * x.data


def test_pipeline():

    class F1(pept.base.LineDataFilter):
        def fit_sample(self, sample_lines):
            sample_lines.lines[:] += 1
            sample_lines.attrs["attr1"] = "New attribute added by F1"
            return sample_lines


    class F2(pept.base.LineDataFilter):
        def fit_sample(self, sample_lines):
            sample_lines.lines[:] += 2
            sample_lines.attrs["attr2"] = "New attribute added by F2"
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
    assert "attr1" in F1().fit_sample(lines[0]).attrs
    assert "attr1" in pept.base.Pipeline([F1()]).fit_sample(lines[0]).attrs

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
    assert "attr1" in F1().fit(lines)[0].attrs
    assert "attr1" in pept.base.Pipeline([F1()]).fit(lines)[0].attrs
