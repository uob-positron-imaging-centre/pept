#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# File   : __init__.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 21.08.2019


'''Tracer location, identification and tracking algorithms.

The `pept.tracking` subpackage hosts different tracking algorithms, working
with both the base classes, as well as with generic NumPy arrays.

All algorithms here are either ``pept.base.Filter`` or ``pept.base.Reducer``
subclasses, implementing the `.fit` and `.fit_sample` methods; here is an
example using PEPT-ML:

>>> from pept.tracking import *
>>>
>>> cutpoints = Cutpoints(0.5).fit(lines)
>>> clustered = HDBSCAN(0.15).fit(cutpoints)
>>> centres = (SplitLabels() + Centroids() + Stack()).fit(clustered)

Once the processing steps have been tuned (see the `Tutorials`), you can chain
all filters into a `pept.Pipeline` for efficient, parallel execution:

>>> pipeline = (
>>>     Cutpoints(0.5) +
>>>     HDBSCAN(0.15) +
>>>     SplitLabels() + Centroids() + Stack()
>>> )
>>> centres = pipeline.fit(lines)

If you would like to implement a PEPT algorithm, all you need to do is to
subclass a ``pept.base.Filter`` and define the method ``.fit_sample(sample)`` -
and you get parallel execution and pipeline chaining for free!

>>> import pept
>>>
>>> class NewAlgorithm(pept.base.LineDataFilter):
>>>     def __init__(self, setting1, setting2 = None):
>>>         self.setting1 = setting1
>>>         self.setting2 = setting2
>>>
>>>     def fit_sample(self, sample: pept.LineData):
>>>         processed_points = ...
>>>         return pept.PointData(processed_points)

'''


from    .birmingham_method      import  BirminghamMethod
from    .peptml                 import  Cutpoints, Minpoints
from    .peptml                 import  HDBSCAN, HDBSCANClusterer
from    .fpi                    import  FPI

from    .transformers           import  Stack

from    .transformers           import  SplitLabels
from    .transformers           import  SplitAll, GroupBy

from    .transformers           import  Centroids
from    .transformers           import  LinesCentroids

from    .transformers           import  Condition
from    .transformers           import  Remove
from    .transformers           import  Swap

from    .space_transformers     import  Voxelize
from    .space_transformers     import  Interpolate
from    .space_transformers     import  Reorient

from    .post                   import  Velocity

from    .tof                    import  TimeOfFlight
from    .tof                    import  CutpointsToF
from    .tof                    import  GaussianDensity

from    .trajectory_separation  import  Segregate
from    .trajectory_separation  import  Reconnect


__license__ = "GNU v3.0"
__maintainer__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__status__ = "Beta"
