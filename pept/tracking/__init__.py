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
All functions from the subpackages are also imported into the `pept.tracking`
root, so you can use both `pept.tracking.fpi.FPI` and `pept.tracking.FPI`
depending on how verbose you want to be.

::

    pept.tracking
    │
    Modules imported into the subpackage root:
    ├── birmingham_method :     The Birmingham method for single-tracers.
    ├── peptml :                The PEPT-ML multi-tracer tracking algorithm.
    ├── fpi :                   The FPI multi-tracer tracking algorithm.
    ├── trajectory_separation : Separate located points into distinct tracks.
    │
    Subpackages:
    ├── birmingham_method
    │   └── BirminghamMethod
    ├── fpi
    │   ├── fpi_ext
    │   └── FPI
    ├── peptml
    │   ├── find_cutpoints
    │   ├── get_cutoffs
    │   ├── Cutpoints
    │   └── HDBSCANClusterer
    └── trajectory_separation
        ├── segregate_trajectories
        ├── connect_trajectories
        └── trajectory_errors

'''


from    .birmingham_method      import  *
from    .peptml                 import  *
from    .fpi                    import  *
from    .trajectory_separation  import  *


__all__ = [
    "birmingham_method",
    "peptml",
    "fpi",
    "trajectory_separation"
]


__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Beta"
