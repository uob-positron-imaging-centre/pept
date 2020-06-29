#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.08.2019


'''Transform data from different PET / PEPT scanner geometries and data formats
into the common base classes.

Summary
-------
The PEPT base classes `PointData`, `LineData`, and `VoxelData` are abstractions
over the type of data that may be encountered in the context of PEPT (e.g. LoRs
are `LineData`, trajectory points are `PointData`), as once the raw data is
transformed into the common formats, any tracking, analysis or visualisation
algorithm in the `pept` package can be used interchangeably.

The `pept.scanners` subpackage provides modules for transforming the raw data
from different PET / PEPT scanner geometries (parallel screens, modular
cameras, etc.) and data formats (binary, ASCII, etc.) into the common base
classes.

If you'd like to integrate another scanner geometry or raw data format into
this package, you can check out the `pept.scanners.parallel_screens` module as
an example. This usually only involves writing a single function by hand; then
all attributes and methods from `LineData` will be available to your new data
format.

Subpackages Provided
--------------------

::

    pept.scanners
    │
    Classes imported into subpackage root:
    ├── ModularCamera :    Convert modular cameras data into `LineData`.
    ├── ParallelScreens :  Initialise parallel screens data as `LineData`.
    │
    Subpackages
    ├── modular_camera
    │   └── ModularCamera
    └── parallel_screens
        └── ParallelScreens

'''


from    .parallel_screens    import  *
from	.modular_camera	     import  *


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


