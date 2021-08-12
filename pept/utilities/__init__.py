#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 14.01.2020


'''PEPT-oriented utility functions.

The utility functions include low-level optimised Cython functions (e.g.
`find_cutpoints`) that are of common interest across the `pept` package, as
well as I/O functions, parallel maps and pixel/voxel traversal algorithms.

Even though the functions are grouped in directories (subpackages) and files
(modules), unlike the rest of the package, they are all imported into the
`pept.utilities` root, so that their import paths are not too long.
'''


from    .cutpoints  import  *
from    .traverse   import  *
from    .parallel   import  *
from    .misc       import  *


__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Beta"
