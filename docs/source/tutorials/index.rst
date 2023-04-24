*********
Tutorials
*********

The main purpose of the PEPT library is to provide a common, consistent
foundation for PEPT-related algorithms, including tracer tracking,
visualisation and post-processing tools - such that they can be used
interchangeably, mixed and matched for any PEPT camera and system. Virtually
all PEPT processing routine follows these steps:

1. Convert raw gamma camera / scanner data into 3D lines (i.e. the captured
   gamma rays, or lines of response - LoRs).
2. Take a sample of lines, locate tracer locations, then repeat for the next
   samples.
3. Separate out individual tracer trajectories.
4. Visualise and post-process trajectories.

For these algorithm-agnostic steps, PEPT provides five base data structures
upon which the rest of the library is built:

1. ``pept.LineData``: general 3D line samples, formatted as *[time, x1, y1, z1,
   x2, y2, z2, extra...]*.
2. ``pept.PointData``: general 3D point samples, formatted as *[time, x, y, z,
   extra...]*.
3. ``pept.Pixels``: single 2D pixellised space with physical dimensions,
   including fast line traversal.
4. ``pept.Voxels``: single 3D voxellised space with physical dimensions,
   including fast line traversal.

For example, once you convert your PEPT data - from any scanner - into
``pept.LineData``, all the algorithms in this library can be used.

All the data structures above are built on top of NumPy and integrate natively
with the rest of the Python / SciPy ecosystem. The rest of the PEPT library is
organised into submodules:

1. ``pept.scanners``: converters between native scanner data and the base
   data structures.
2. ``pept.tracking``: radioactive tracer tracking algorithms, e.g. the
   Birmingham method, PEPT-ML, FPI.
3. ``pept.plots``: PEPT data visualisation subroutines.
4. ``pept.utilities``: general-purpose helpers, e.g. ``read_csv``,
   ``traverse3d``.
5. ``pept.processing``: PEPT-oriented post-processing algorithms, e.g.
   ``VectorField3D``.


------------


If you are new to the PEPT library, we recommend going through this interactive
online notebook, which introduces all the fundamental concepts of the library:

    https://colab.research.google.com/drive/1G8XHP9zWMMDVu23PXzANLCOKNP_RjBEO?usp=sharing


Once you get the idea of ``LineData`` samples, ``Pipeline`` and
``PlotlyGrapher``, you can use these copy-pastable tutorials to build PEPT data
analysis pipelines tailored to your specific systems.


.. toctree::
   :caption: Pre-processing

   basics
   reading
   visualising
   converting



.. toctree::
   :caption: Tracking

   adaptive_samples
   birmingham
   peptml
   fpi



.. toctree::
   :caption: Post-processing

   tracking_errors
   trajectory_separation
   filtering
   velocities
   interpolating


