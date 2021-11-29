#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License:  GNU v3.0
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
'''


from    .trajectory_separation      import  Segregate
from    .trajectory_separation      import  Reconnect


__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Beta"
