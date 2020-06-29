#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.06.2020


'''The PEPT cookbook, containing example analysis scripts (or recipes) that are
easy to hack into.

Summary
-------
Following the cookbook metaphor, this package contains a collection of scripts
combining different parts of the `pept` library (i.e. ingredients) to
accomplish a certain task -> recipes! Much like when cooking, it is more than
recommended to take the recipes (i.e. code) and make them yours; they are
well-commented and user-friendly.

Extended Summary
----------------
The recipes are implemented *as classes*, such that once they've been run (i.e.
instantiated), all the relevant objects (such as clusterers, graphers, etc.)
can be accessed as class attributes.

Modules Provided
----------------

::

    pept.cookbook
    │
    Classes imported into the subpackage root:
    ├── PEPTMLUser :            Use PEPT-ML to turn LoRs into trajectories.
    └── PEPTMLFindParameters :  Visual aid for PEPT-ML clustering parameters.

Notes
-----
At the moment, the subpackages in `pept` are biased towards PEPT-ML, as there
aren't many algorithms integrated into package *yet*. New algorithms and/or
recommendations for the package are more than welcome! `pept` aims to be a
community effort, be it academic, industrial, medical, or just from PEPT
enthusiasts - so it is open to help with documentation, algorithms, utilities
or analysis scripts, tutorials, and pull requests in general!
'''


from    .peptml     import  PEPTMLUser
from    .peptml     import  PEPTMLFindParameters


__all__ = [
    "PEPTMLUser",
    "PEPTMLFindParameters"
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


