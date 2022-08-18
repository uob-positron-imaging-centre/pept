Absolute Basics
===============

The main purpose of the `pept` library is to provide a common, consistent foundation for PEPT-related algorithms, including tracer tracking, visualisation and post-processing tools - such that they can be used interchangeably, mixed and matched for different systems. Virtually *any* PEPT processing routine follows these steps:

1. Convert raw gamma camera / scanner data into *3D lines* (i.e. the captured gamma rays, or lines of response - LoRs).
2. Take a *sample* of lines, locate tracer locations, then repeat for the next samples.
3. Separate out individual tracer trajectories.
4. Visualise and post-process trajectories.

For these algorithm-agnostic steps, `pept` provides five base data structures upon which the rest of the library is built:

1. [`pept.LineData`](https://pept.readthedocs.io/en/latest/manual/generated/pept.LineData.html): general 3D line samples, formatted as *[time, x1, y1, z1, x2, y2, z2, extra...]*.
2. [`pept.PointData`](https://pept.readthedocs.io/en/latest/manual/generated/pept.PointData.html): general 3D point samples, formatted as *[time, x, y, z, extra...]*.
3. [`pept.Pixels`](https://pept.readthedocs.io/en/latest/manual/generated/pept.Pixels.html): single 2D pixellised space with physical dimensions, including fast line traversal.
4. [`pept.Voxels`](https://pept.readthedocs.io/en/latest/manual/generated/pept.Voxels.html): single 3D voxellised space with physical dimensions, including fast line traversal.

All the data structures above are built on top of NumPy and integrate natively with the rest of the Python / SciPy ecosystem. The rest of the `pept` library is organised into submodules:

- [`pept.scanners`](https://pept.readthedocs.io/en/latest/manual/scanners.html): converters between native scanner data and the base classes.
- [`pept.tracking`](https://pept.readthedocs.io/en/latest/manual/tracking.html): radioactive tracer tracking algorithms, e.g. the Birmingham method, PEPT-ML, FPI.
- [`pept.plots`](https://pept.readthedocs.io/en/latest/manual/plots.html): PEPT data visualisation subroutines.
- [`pept.utilities`](https://pept.readthedocs.io/en/latest/manual/utilities.html): general-purpose helpers, e.g. `read_csv`, `traverse3d`.
- [`pept.processing`](https://pept.readthedocs.io/en/latest/manual/processing.html): PEPT-oriented post-processing algorithms, e.g. `occupancy2d`.


[`pept.LineData`](https://pept.readthedocs.io/en/latest/manual/generated/pept.LineData.html)
--------------------------------------------------------------------------------------------

Generally, PEPT Lines of Response (LoRs) are lines in 3D space, each
defined by two points, regardless of the geometry of the scanner used. This
class is used to wrap LoRs (or any lines!), efficiently yielding samples of
`lines` of an adaptive `sample_size` and `overlap`.

It is an abstraction over PET / PEPT scanner geometries and data formats,
as once the raw LoRs (be they stored as binary, ASCII, etc.) are
transformed into the common `LineData` format, any tracking, analysis or
visualisation algorithm in the `pept` package can be used interchangeably.
Moreover, it provides a stable, user-friendly interface for iterating over
LoRs in *samples* - this is useful for tracking algorithms, as they
generally take a few LoRs (a *sample*), produce a tracer position, then
move to the next sample of LoRs, repeating the procedure. Using overlapping
samples is also useful for improving the tracking rate of the algorithms.

Here are some basic examples of creating and using `LineData` samples - you're
very much invited to copy and run them!

Initialise a `LineData` instance containing 10 lines with a `sample_size`
of 3.

```python
>>> import pept
>>> import numpy as np
>>> lines_raw = np.arange(70).reshape(10, 7)
>>> print(lines_raw)
[[ 0  1  2  3  4  5  6]
 [ 7  8  9 10 11 12 13]
 [14 15 16 17 18 19 20]
 [21 22 23 24 25 26 27]
 [28 29 30 31 32 33 34]
 [35 36 37 38 39 40 41]
 [42 43 44 45 46 47 48]
 [49 50 51 52 53 54 55]
 [56 57 58 59 60 61 62]
 [63 64 65 66 67 68 69]]

>>> line_data = pept.LineData(lines_raw, sample_size = 3)
>>> line_data
pept.LineData (samples: 3)
--------------------------
sample_size = 3
overlap = 0
lines =
  (rows: 10, columns: 7)
  [[ 0.  1. ...  5.  6.]
   [ 7.  8. ... 12. 13.]
   ...
   [56. 57. ... 61. 62.]
   [63. 64. ... 68. 69.]]
columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
attrs = {}
```

Access samples using subscript notation. Notice how the samples are
consecutive, as `overlap` is 0 by default.

```python
>>> line_data[0]
pept.LineData (samples: 1)
--------------------------
sample_size = 3
overlap = 0
lines =
  (rows: 3, columns: 7)
  [[ 0.  1. ...  5.  6.]
   [ 7.  8. ... 12. 13.]
   [14. 15. ... 19. 20.]]
columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
attrs = {}

>>> line_data[1]
pept.LineData (samples: 1)
--------------------------
sample_size = 3
overlap = 0
lines =
  (rows: 3, columns: 7)
  [[21. 22. ... 26. 27.]
   [28. 29. ... 33. 34.]
   [35. 36. ... 40. 41.]]
columns = ['t', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2']
attrs = {}
```

Now set an overlap of 2; notice how the number of samples changes:

```python
>>> len(line_data)     # Number of samples
3

>>> line_data.overlap = 2
>>> len(line_data)
8
```




