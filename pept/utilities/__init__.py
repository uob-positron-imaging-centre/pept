#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 14.01.2020


'''PEPT-oriented utility functions.

Summary
-------
The utility functions include low-level optimised Cython functions (e.g.
`find_cutpoints`) that are of common interest across the `pept` package, as
well as I/O functions, parallel maps and pixel/voxel traversal algorithms.

Even though the functions are grouped in directories (subpackages) and files
(modules), unlike the rest of the package, they are all imported into the
`pept.utilities` root, so that their import paths are not too long.

Subpackages Provided
--------------------

::

    pept.utilities
    │
    Functions imported into the subpackage root:
    ├── find_cutpoints :            Find cutpoints from an array of LoRs.
    ├── group_by_columns :          Group array into a list by column.
    ├── number_of_lines :           Find the number of lines in a file.
    ├── read_csv :                  Fast CSV file reader.
    ├── read_csv_chunks :           Read CSV file in chunks using a generator.
    ├── parallel_map_file :         Map a function to chunks of a file.
    ├── traverse2d :                Traverse pixels.
    ├── traverse3d :                Traverse voxels.
    │
    Classes imported into the subpackage root:
    ├── ChunkReader :               Lazily read/access chunks from a CSV file.
    │
    Subpackages
    ├── cutpoints
    ├── misc
    ├── parallel
    └── traverse

'''


from    .cutpoints  import  *
from    .traverse   import  *
from    .parallel   import  *
from    .misc       import  *


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


