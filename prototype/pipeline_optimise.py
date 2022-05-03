#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_optimise.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 24.11.2021


import numpy as np
import pept
from pept.tracking import *
from pept.plots import PlotlyGrapher
from pept.plots import PlotlyGrapher2D

import plotly.graph_objs as go
from plotly.subplots import make_subplots


lors = pept.scanners.parallel_screens(
    "https://raw.githubusercontent.com/uob-positron-imaging-centre/" +
    "example_data/master/sample_1p_fluidised_bed.csv",
    600,
    sample_size = 200,
    skiprows = 15,
)


pipeline = pept.Pipeline([
    Stack(sample_size = 300, overlap = 150),
    BirminghamMethod(fopt = 0.5),
    Stack(),
])


# Create PEPT-ML processing pipeline
pipeline = pept.Pipeline([
    # First pass of clustering
    Stack(sample_size = 136, overlap = 68),
    Cutpoints(max_distance = 0.4),
    HDBSCAN(true_fraction = 0.96),
    SplitLabels() + Centroids(error = True),
    Stack(),
])


hist = pipeline.optimise(
    lors.lines[:10],
    sample_size = [100, 200],
    overlap = [0, 190],
    max_distance = [0.01, 2.0],
    true_fraction = [0, 1],
)


nanhist = ~np.isfinite(hist[:, -1])
smahist = hist[:, -1] < np.quantile(hist[:, -1], 0.8)

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x = hist[smahist, 0],
    y = hist[smahist, 1],
    z = hist[smahist, -1],
    mode = "markers",
    marker = dict(
        color = np.arange(smahist.sum()),
    )
))

fig.add_trace(go.Scatter3d(
    x = hist[nanhist, 0],
    y = hist[nanhist, 1],
    z = np.zeros(nanhist.sum()),
    mode = "markers",
    marker = dict(
        color = "red",
    )
))

fig.show()


fig2 = make_subplots(3, 1)
ep = np.arange(len(hist))
fig2.add_trace(go.Scatter(x=ep, y = hist[:, -3]), 1, 1)
fig2.add_trace(go.Scatter(x=ep, y = hist[:, -2]), 2, 1)
fig2.add_trace(go.Scatter(x=ep, y = hist[:, -1]), 3, 1)
fig2.show()


traj = pipeline.fit(lors)


PlotlyGrapher().add_points(traj).show()
PlotlyGrapher2D().add_timeseries(traj).show()
