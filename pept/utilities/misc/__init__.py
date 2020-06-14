#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 10.06.2020


from    .aggregate  import  group_by_column
from    .read_csv   import  number_of_lines
from    .read_csv   import  read_csv
from    .read_csv   import  read_csv_chunks
from    .read_csv   import  ChunkReader


__all__ = [
    "group_by_column",
    "number_of_lines",
    "read_csv",
    "read_csv_chunks",
    "ChunkReader"
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


