#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.08.2019


'''The `peptml` package implements a hierarchical density-based clustering
algorithm for general Positron Emission Particle Tracking (PEPT).

The PEPT-ML algorithm [1] works using the following steps:
    1. Split the data into a series of individual "samples", each containing
    a given number of LoRs. Use the base class pept.LineData for this.
    2. For every sample of LoRs, compute the *cutpoints*, or the points in
    space that minimise the distance to every pair of lines.
    3. Cluster every sample using HDBSCAN and extract the centres of the
    clusters ("1-pass clustering").
    4. Split the centres into samples of a given size.
    5. Cluster every sample of centres using HDBSCAN and extract the centres
    of the clusters ("2-pass clustering").
    6. Construct the trajectory of every particle using the centres from the
    previous step.

A typical workflow for using the `peptml` package would be:
    1. Read the LoRs into a `pept.LineData` class instance and set the
    `sample_size` and `overlap` appropriately.
    2. Compute the cutpoints using the `pept.tracking.peptml.Cutpoints` class.
    3. Instantiate an `pept.tracking.peptml.HDBSCANClusterer` class and cluster
    the cutpoints found previously.

More tutorials and examples can be found on the University of Birmingham
Positron Imaging Centre's GitHub repository.

PEPT-ML was successfuly used at the University of Birmingham to analyse real
Fluorine-18 tracers in air.

.. [1] Nicuşan AL, Windows-Yule CR. Positron emission particle tracking
   using machine learning. Review of Scientific Instruments.
   2020 Jan 1;91(1):013329.
   https://doi.org/10.1063/1.5129251

'''


from    .cutpoints  import  find_cutpoints
from    .cutpoints  import  get_cutoffs

from    .cutpoints  import  find_cutpoints_tof
from    .cutpoints  import  get_cutoffs_tof

from    .cutpoints  import  Cutpoints
from    .cutpoints  import  CutpointsToF

from    .peptml     import  HDBSCANClusterer


__all__ = [
    "find_cutpoints",
    "get_cutoffs",
    "find_cutpoints_tof",
    "get_cutoffs_tof"
    "Cutpoints",
    "CutpointsToF"
    "HDBSCANClusterer"
]


__author__ = "Andrei Leonard Nicusan"
__credits__ = [
    "Andrei Leonard Nicusan",
    "Kit Windows-Yule",
    "Sam Manger"
]
__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Development"


