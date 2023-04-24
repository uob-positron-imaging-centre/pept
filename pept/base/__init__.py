#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 31.01.2020


'''PEPT base classes.
'''


from    .line_data          import  LineData
from    .point_data         import  PointData

from    .iterable_samples   import  IterableSamples
from    .iterable_samples   import  TimeWindow
from    .iterable_samples   import  AdaptiveWindow
from    .iterable_samples   import  AsyncIterableSamples
from    .iterable_samples   import  PEPTObject

from    .pipelines          import  Transformer
from    .pipelines          import  Filter
from    .pipelines          import  LineDataFilter
from    .pipelines          import  PointDataFilter
from    .pipelines          import  VoxelsFilter
from    .pipelines          import  Reducer
from    .pipelines          import  Pipeline

from    .utilities          import  check_iterable


# Execute code here to add dynamic methods
from    .pixels             import  Pixels
from    .voxels             import  Voxels




__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Beta"
