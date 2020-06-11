#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.08.2019


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


