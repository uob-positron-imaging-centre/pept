#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 21.08.2019


'''Tracer location, identification and tracking algorithms.

Summary
-------
The `pept.tracking` subpackage hosts different tracking algorithms, working
with both the base classes, as well as with generic NumPy arrays. The ones that
use the base classes provide parallelised subroutines, being faster and easier
to use, while the functions using NumPy arrays provide fine-grained control
over the individual samples of lines / points being used.

Subpackages Provided
--------------------

::

    pept.tracking
    │
    Modules imported into the subpackage root:
    ├── birmingham_method :     The Birmingham method for single-tracers.
    ├── peptml :                The PEPT-ML multi-tracer tracking algorithm.
    ├── trajectory_separation : Separate located points into distinct tracks.
    │
    Subpackages:
    ├── birmingham_method
    │   └── BirminghamMethod
    ├── peptml
    │   ├── find_cutpoints
    │   ├── get_cutoffs
    │   ├── Cutpoints
    │   └── HDBSCANClusterer
    └── trajectory_separation
        ├── segregate_trajectories
        ├── connect_trajectories
        └── trajectory_errors

Notes
-----
At the moment, the subpackages in `pept.tracking` are biased towards PEPT-ML,
as there aren't many algorithms integrated into package *yet*. New algorithms
and/or recommendations for the package are more than welcome! `pept` aims to be
a community effort, be it academic, industrial, medical, or just from PEPT
enthusiasts - so it is open to help with documentation, algorithms, utilities
or analysis scripts, tutorials, and pull requests in general!
'''


from    .   import  peptml
from    .   import  birmingham_method
from    .   import  trajectory_separation


__all__ = [
    "peptml",
    "birmingham_method",
    "trajectory_separation"
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


