..
   File   : pept_api.rst
   License: GNU v3.0
   Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
   Date   : 29.06.2020


PEPT Library API
================

.. automodule:: pept


Base Classes
------------

LineData
^^^^^^^^
.. autoclass:: pept.LineData
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__


PointData
^^^^^^^^^
.. autoclass:: pept.PointData
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __init__


Voxels
^^^^^^
.. autoclass:: pept.Voxels
   :members: add_lines, get_cutoff, cube_trace, cubes_traces, heatmap_trace
   :show-inheritance:
   :special-members: __new__


Subpackages
-----------

.. toctree::
   :maxdepth: 3

   api/pept.cookbook
   api/pept.diagnostics
   api/pept.scanners
   api/pept.tracking
   api/pept.utilities
   api/pept.visualisation


