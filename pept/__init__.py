#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 19.08.2019


'''
PEPT
====

Summary
-------
A Python library that unifies Positron Emission Particle Tracking (PEPT)
research, including tracking, simulation, data analysis and visualisation
tools.

Extended Summary
----------------
This Python library integrates all the tools necessary to perform research
using Positron Emission Particle Tracking (PEPT). The library includes
algorithms for the location, identification and tracking of particles, in
addition to tools for visualisation and analysis, and utilities allowing the
realistic simulation of PEPT data.

PEPT is a technique developed at the University of Birmingham [1]_ which allows
the non-invasive, three-dimensional tracking of one or more 'tracer' particles
through particulate, fluid or multiphase systems. The technique allows particle
or fluid motion to be tracked with sub-millimetre accuracy and sub-millisecond
temporal resolution and, due to its use of highly-penetrating 511keV gamma
rays, can be used to probe the internal dynamics of even large, dense,
optically opaque systems - making it ideal for industrial as well as
scientific applications.

PEPT is performed by radioactively labelling a particle with a positron-
emitting radioisotope such as fluorine-18 (18F) or gallium-66 (66Ga), and using
the back-to-back gamma rays produced by electron-positron annihilation events
in and around the tracer to triangulate its spatial position. Each detected
gamma ray represents a line of response (LoR) .

Subpackages Provided
--------------------
The `pept` package provides three base classes at its root - `PointData`,
`LineData`, and `VoxelData` - acting as common data formats that the rest of
the package will use for its tracking, analysis and visualisation algorithms.
Thus any subroutine integrated in the `pept` package can be used
interchangeably with new PET / PEPT scanner geometries, data formats or novel
algorithms.

The rest of the package is grouped into subpackages and modules following the
hierarchy below:

::

    pept
    │
    Base classes imported into the package root:
    ├── PointData :     Base class encapsulating points.
    ├── LineData :      Base class encapsulating lines (LoRs) w/ one timestamp.
    ├── VoxelData :     Base class encapsulating voxels.
    │
    Subpackages:
    ├── base :                  Base classes (above).
    ├── cookbook :              Pre-made PEPT analysis scripts, or recipes.
    ├── diagnostics :           PET/PEPT scanner diagnostics.
    ├── scanners :              Transform other data formats into base classes.
    │   ├── modular_camera :    Birmingham modular cameras binary data.
    │   └── parallel_screens :  Birmingham parallel screens PEPT detector.
    ├── simulation :            Simulate radioactively-labeled tracers.
    ├── tests :                 Package unit tests.
    ├── tracking :              Tracer identification and tracking algorithms.
    │   ├── birmingham_method : The original Birmingham Method [1].
    │   ├── peptml :            The PEPT-ML algorithm [2].
    │   └── trajectory_separation : Trajectory separation of tracked tracers.
    ├── utilities :             Utility functions such as fast CSV-readers.
    │   ├── cutpoints :         Compute cutpoints from LoRs; Cython functions.
    │   ├── misc :              Miscellaneous, I/O, data aggregation.
    │   ├── parallel :          Call arbitrary functions using multithreading.
    │   └── traverse :          Traverse voxels for LoRs; Cython functions.
    └── visualisation :         Visualisation algorithms for LoRs, points, etc.


Each subpackage has its own documentation containing module, class and function
hierarchies such as the one above.

Performance
-----------
Significant effort has been put into making the algorithms in this package as
fast as possible. The most compute-intensive parts have been implemented in
`C` / `Cython` and parallelised, where possible, using `joblib` and
`concurrent.futures.ThreadPoolExecutor`. For example, using the `peptml`
subpackage [2]_, analysing 1,000,000 LoRs on the author's machine (mid 2012
MacBook Pro) takes ~26 seconds.

Citing
------
If you used this codebase or any software making use of it in a scientific
publication, we ask you to cite the following paper:

    Nicuşan AL, Windows-Yule CR. Positron emission particle tracking using
    machine learning. Review of Scientific Instruments. 2020 Jan 1;91(1):013329
    https://doi.org/10.1063/1.5129251

Licensing
---------
The `pept` package is GNU v3.0 licensed.
Copyright (C) 2020 Andrei Leonard Nicusan.

References
----------
.. [1] Parker DJ, Broadbent CJ, Fowles P, Hawkesworth MR, McNeil P. Positron
   emission particle tracking-a technique for studying flow within engineering
   equipment. Nuclear Instruments and Methods in Physics Research Section A:
   Accelerators, Spectrometers, Detectors and Associated Equipment. 1993
   Mar 10;326(3):592-607.
.. [2] Nicuşan AL, Windows-Yule CR. Positron emission particle tracking using
   machine learning. Review of Scientific Instruments. 2020 Jan 1;91(1):013329.

Examples
--------
You can download data samples from the UoB Positron Imaging Centre's
Repository (`link <https://bit.ly/pept-example-data-repo>`_). A
small but complete analysis script using the PEPT-ML algorithm is given below.

First import the `pept` package and read in lines of response (LoRs) from our
online repository. We'll use a sample of real data from an experiment conducted
at Birmingham - two tracers rotating at 42RPM.

.. code-block:: python

    import pept

    lors_raw = pept.utilities.read_csv(
        ("https://raw.githubusercontent.com/uob-positron-imaging-centres/"
         "example_data/master/sample_2p_42rpm.csv"),  # Concatenate long string
        skiprows = 16                                 # Skip file header
    )

Our data is in a format typical of the parallel screens PEPT detector from
Birmingham. Convert it to our package's scanner-agnostic line format; we'll
iterate through them one sample at a time, tracking the tracer locations:

.. code-block:: python

    lors = pept.scanners.ParallelScreens(
        lors_raw, screen_separation = 712, sample_size = 200
    )

Now it's time for some machine learning! Import the `peptml` package and
transform the LoRs into *cutpoints* that we'll use to find tracer locations.

.. code-block:: python

    from pept.tracking import peptml
    cutpoints = peptml.Cutpoints(lors, max_distance = 0.15)

Create a *clusterer* and use the cutpoints to find the centres of the tracers
as they move through our system:

.. code-block:: python

    clusterer = peptml.HDBSCANClusterer()
    centres = clusterer.fit(cutpoints)

This data looks alright (we'll plot everything in a minute), but we can do
better. We can iterate through the centres once more, clustering them again,
to get smooth, tight trajectories; for this, we'll set a small sample size and
large overlap:

.. code-block:: python

    centres.sample_size = 60
    centres.overlap = 59
    centres_2pass = clusterer.fit(centres)

One more step! The locations of the two tracers tracked above are intertwined;
we need to separate them into individual trajectories. We have a module for
just that:

.. code-block:: python

    import pept.tracking.trajectory_separation as tsp
    points_window = 10
    trajectory_cut_distance = 20
    trajectories = tsp.segregate_trajectories(
        centres_2pass, points_window, trajectory_cut_distance
    )

Finally, let's plot some LoRs, the tracer centres after one pass of clustering
and after the second pass of clustering, and the separated trajectories. Let's
use Plotly to produce some beautiful, interactive, 3D graphs:

.. code-block:: python

    from pept.visualisation import PlotlyGrapher
    subplot_titles = ["Lines of Response (LoRs)", "HDBSCAN Clustering",
                      "2-pass Clustering", "Separated Trajectories"]
    grapher = PlotlyGrapher(rows = 2, cols = 2, zlim = [0, 712],
                            subplot_titles = subplot_titles)

    grapher.add_lines(lors.lines[:400])             # Only the first 400 LoRs
    grapher.add_points(centres, col = 2)
    grapher.add_points(centres_2pass, row = 2)
    grapher.add_points(trajectories, row = 2, col = 2)

    grapher.show()

A more in-depth tutorial is available on
`Google Colab <https://bit.ly/pept-tutorial-1>`_.

'''


# Import base data structures
from    .base.line_data     import  LineData
from    .base.point_data    import  PointData
from    .base.voxel_data    import  VoxelData

# Import subpackages
from    .                   import  cookbook
from    .                   import  diagnostics
from    .                   import  scanners
from    .                   import  simulation
from    .                   import  tracking
from    .                   import  utilities
from    .                   import  visualisation

# Import package version
from    .__version__        import  __version__


__all__ = [
    'LineData',
    'PointData',
    'VoxelData',
    'cookbook',
    'diagnostics',
    'scanners',
    'simulation',
    'tracking',
    'utilities',
    'visualisation'
]


__author__ = ["Andrei Leonard Nicusan", "Sam Manger"]
__credits__ = [
    "Andrei Leonard Nicusan",
    "Kit Windows-Yule",
    "Sam Manger"
]
__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Development"


