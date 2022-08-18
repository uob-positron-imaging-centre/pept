Tracking Errors
===============

When processing more difficult datasets - scattering environments, low tracer activities, etc. -
it is often useful to use some tracer statistics to remove erroneous locations.

Most PEPT algorithms will include some measure of the tracer location errors, for example:

- The ``Centroids(error = True)`` filter appends a column "error" representing the standard
  deviation of the distances from the computed centroid to the constituent points. For a
  500 mm scanner, a spread in a tracer location of 100 mm is clearly an erroneous point.
- The ``Centroids(cluster_size = True)`` filter appends a column "cluster_size" representing
  the number of points used to compute the centroid. If a sample of 200 LoRs yields a tracer
  location computed from 5 points, it is clearly noise.
- The ``BirminghamMethod`` filter includes a column "error" representing the standard
  deviation of the distances from the tracer position to the constituent LoRs.


Histogram of Tracking Errors
----------------------------

You can select a named column via string indexing, e.g.  ``trajectories["error"]``; you can
then plot a histogram of the relative errors with:

::

    import plotly.express as px
    px.histogram(trajectories["error"]).show()          # Large values are noise
    px.histogram(trajectories["cluster_size"]).show()   # Small values are noise


It is often useful to remove points with an error higher than a certain value, e.g. 20 mm:

::

    trajectories = Condition("error < 20").fit(trajectories)

    # Or simply append the `Condition` to the `pept.Pipeline`
    pipeline = pept.Pipeline([
        ...
        Condition("cluster_size > 30, error < 20"),
        ...
    ])


