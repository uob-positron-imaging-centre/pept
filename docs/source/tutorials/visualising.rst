Plotting
========



Interactive 3D Plots
--------------------

The easiest method of plotting 3D PEPT-like data is using the ``pept.plots.PlotlyGrapher``
interactive grapher:


::

    # Plotting some example 3D lines
    import pept
    from pept.plots import PlotlyGrapher
    import numpy as np

    lines_raw = np.arange(70).reshape((10, 7)
    lines = pept.LineData(lines_raw)

    PlotlyGrapher().add_lines(lines).show()


::

    # Plotting some example 3D points
    import pept
    from pept.plots import PlotlyGrapher
    import numpy as np

    points_raw = np.arange(40).reshape((10, 4)
    points = pept.PointData(points_raw)

    PlotlyGrapher().add_points(points).show()


The ``PlotlyGrapher`` object allows straightforward subplots creation:


::

    # Plot the example 3D lines and points on separate subplots
    grapher = PlotlyGrapher(cols = 2)

    grapher.add_lines(lines)                        # col = 1 by default
    grapher.add_points(points, col = 2)

    grapher.show()


::

    # Plot the example 3D lines and points on separate subplots
    grapher = PlotlyGrapher(rows = 2, cols = 2)

    grapher.add_lines(lines, col = 2)               # row = 1 by default
    grapher.add_points(points, row = 2, col = 2)

    grapher.show()




Histogram of Tracking Errors
----------------------------

The ``Centroids(error = True)`` filter appends a column "error" representing the relative error
in the tracked position. You can select a named column via indexing, e.g. ``trajectories["error"]``;
you can then plot a histogram of the relative errors with:

::

    import plotly.express as px
    px.histogram(trajectories["error"]).show()


It is often useful to remove points with an error higher than a certain value, e.g. 20 mm:

::

    trajectories = Condition("error < 20").fit(trajectories)

    # Or simply append the `Condition` to the `pept.Pipeline`
    pipeline = pept.Pipeline([
        ...
        Condition("error < 20"),
        ...
    ])




Exporting Plotly Graphs as Images
---------------------------------

The standard output of the Plotly grapher is an interactive HTML webpage; however, this can lead to large file sizes or memory overflows. Plotly allows for graphs to be exported as images to alleviate some of these issues.

Ensure you have imported:

::

    import plotly.express as px
    import kaleido
    import plotly.io as pio


There are two main ways of exporting as images:

::

    # Save the inner plotly.Figure attribute of a `grapher`
    # Format can be changed to other image formats
    # Width and height can be adjusted to give the desired image size
    pio.write_image(grapher.fig, filepath, format="png", width=2560, height=1440)

::

    grapher.fig.write_image("figure.png")
