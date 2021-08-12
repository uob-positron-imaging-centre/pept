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
