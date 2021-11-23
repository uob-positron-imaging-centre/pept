#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_processing.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 23.11.2021


'''Integration tests to ensure the `pept` base classes behave correctly and
offer consistent interfaces.
'''


import numpy as np
import pept

from pept.processing import *



def test_dynamic_probability2d():
    # Generate tracer locations
    num_particles = 10
    positions = pept.PointData(
        np.random.uniform(0, 500, (num_particles, 5)),
        columns = ["t", "x", "y", "z", "v"]
    )

    # Test different uses
    pixels = DynamicProbability2D(1., "v", "xy").fit(positions)
    assert pixels.pixels.any(), "all pixels are zero!"

    DynamicProbability2D(0.1, "t", "yz").fit(positions)
    DynamicProbability2D(0.1, 4, "xy").fit(positions)
    DynamicProbability2D(0.1, "v", "xy", xlim = [0, 500]).fit(positions)
    DynamicProbability2D(0.1, "v", "xy", resolution = [20, 20]).fit(positions)
    DynamicProbability2D(0.1, 4, "xy", max_workers = 1).fit(positions)


def test_residence_distribution2d():
    # Generate tracer locations
    num_particles = 10
    positions = pept.PointData(
        np.random.uniform(0, 500, (num_particles, 5)),
        columns = ["t", "x", "y", "z", "v"]
    )

    # Test different uses
    pixels = ResidenceDistribution2D(1., "v").fit(positions)
    assert pixels.pixels.any(), "all pixels are zero!"

    ResidenceDistribution2D(0.1, "t", "yz").fit(positions)
    ResidenceDistribution2D(0.1, 0, "xy").fit(positions)
    ResidenceDistribution2D(0.1, xlim = [0, 500]).fit(positions)
    ResidenceDistribution2D(0.1, resolution = [20, 20]).fit(positions)
    ResidenceDistribution2D(0.1, 0, "xy", max_workers = 1).fit(positions)


def test_dynamic_probability3d():
    # Generate tracer locations
    num_particles = 10
    positions = pept.PointData(
        np.random.uniform(0, 500, (num_particles, 5)),
        columns = ["t", "x", "y", "z", "v"]
    )

    # Test different uses
    voxels = DynamicProbability3D(1., "v").fit(positions)
    assert voxels.voxels.any(), "all voxels are zero!"

    DynamicProbability3D(0.1, "t", "yzx").fit(positions)
    DynamicProbability3D(0.1, 4,).fit(positions)
    DynamicProbability3D(0.1, "v", xlim = [0, 500]).fit(positions)
    DynamicProbability3D(0.1, "v", resolution = [20, 20, 20]).fit(positions)
    DynamicProbability3D(0.1, 4, max_workers = 1).fit(positions)


def test_residence_distribution3d():
    # Generate tracer locations
    num_particles = 10
    positions = pept.PointData(
        np.random.uniform(0, 500, (num_particles, 5)),
        columns = ["t", "x", "y", "z", "v"]
    )

    # Test different uses
    voxels = ResidenceDistribution3D(1., "v").fit(positions)
    assert voxels.voxels.any(), "all voxels are zero!"

    ResidenceDistribution3D(0.1, "t", "yzx").fit(positions)
    ResidenceDistribution3D(0.1, 0).fit(positions)
    ResidenceDistribution3D(0.1, xlim = [0, 500]).fit(positions)
    ResidenceDistribution3D(0.1, resolution = [20, 20, 20]).fit(positions)
    ResidenceDistribution3D(0.1, 0, max_workers = 1).fit(positions)
