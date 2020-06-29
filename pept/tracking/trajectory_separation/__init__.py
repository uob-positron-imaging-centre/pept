#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.08.2019


'''Separate the intertwined points from pre-tracked tracer locations into
individual trajectories.

Extended Summary
----------------
A typical PEPT workflow would involve transforming LoRs into points using some
tracking algorithm. These points include all tracers moving through the system,
being intertwined (e.g. for two tracers A and B, the `point_data` array might
have two entries for A, followed by three entries for B, then one entry for A,
etc.). The points can be segregated based on distance alone using the
`segregate_trajectories` function; for well-defined trajectories of tracers
that do not collide, this may be enough to retrieve individual trajectories.
However, for tracers that do come into contact, the identity of the least
active one is usually lost; in such cases, the `connect_trajectories` function
can be used to piece back the trajectories of tracers with gaps in their tracks
using some *tracer signature* (e.g. cluster size in PEPT-ML).

Functions Provided
------------------

::

    pept.tracking.trajectory_separation
    ├── segregate_trajectories : Segregate intertwined points by distance.
    ├── connect_trajectories : Connect segregated paths by tracer signatures.
    └── trajectory_errors : Calculate deviations of tracked from real paths.

Examples
--------
Take for example two tracers that go downwards (below, 'x' is the position, and
in parantheses is the array index at which that point is found in the data
array).

::

                    Some tracking algorithm
                    -----------------------
                               |
                               V

    `points`, numpy.ndarray, shape (11, 4), columns [time, x, y, z]:
         x (0)
        x (1)                       x (2)
         x (3)                     x (4)
           x (5)                 x (7)
           x (6)                x (9)
          x (8)                 x (10)

                  Use `segregate_trajectories`
                  ----------------------------
                               |
                               V

>>> import pept.tracking.trajectory_separation as tsp
>>> points_window = 10
>>> trajectory_cut_distance = 15    # mm
>>> segregated_trajectories = tsp.segregate_trajectories(
>>>     points, points_window, trajectory_cut_distance
>>> )

::

                Labelled, segregated trajectories
                ---------------------------------
                               |
                               V

    `segregated_trajectories`, numpy.ndarray, shape (11, 5),
    columns [time, x, y, z, trajectory_label]:
         x (0, label = 0)
        x (1, label = 0)            x (2, label = 1)
         x (3, label = 0)          x (4, label = 1)
           x (5, label = 0)      x (7, label = 1)
           x (6, label = 0)     x (9, label = 1)
          x (8, label = 0)      x (10, label = 1)

The usage of `connect_trajectories` is better explained using visual aids.
Check out the tutorials online at the Birmingham Positron Imaging Centre's
GitHub repository - https://github.com/uob-positron-imaging-centre.
'''


from    .trajectory_separation      import  segregate_trajectories
from    .trajectory_separation      import  connect_trajectories
from    .trajectory_separation      import  trajectory_errors


__all__ = [
    "segregate_trajectories",
    "connect_trajectories",
    "trajectory_errors"
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


