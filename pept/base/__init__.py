#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 31.01.2020


'''PEPT base classes.

Subpackages Provided
--------------------
These classes are imported into the ``pept`` package root, so they are
available as ``pept.PointData``, ``pept.LineData``, etc., without going through
``base`` first.

::

    pept.base
    ├── PointData :     Encapsulate points for iteration and visualisation.
    ├── LineData :      Encapsulate lines (LoRs) with a single timestamp.
    ├── Pixels :        Encapsulate pixels for a single sample of lines.
    ├── Voxels :        Encapsulate voxels for a single sample of lines.
    └── VoxelData :     Encapsulate voxels for line traversal and manipulation.

'''


from    .line_data      import  LineData
from    .point_data     import  PointData
from    .voxel_data     import  VoxelData
from    .voxel_data     import  Voxels
from    .pixel_data     import  Pixels


__all__ = [
    'LineData',
    'PointData',
    'Pixels',
    'Voxels',
    'VoxelData',
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
__status__ = "Stable"
