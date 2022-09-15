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

    lines_raw = np.arange(70).reshape((10, 7))
    lines = pept.LineData(lines_raw)

    PlotlyGrapher().add_lines(lines).show()


::

    # Plotting some example 3D points
    import pept
    from pept.plots import PlotlyGrapher
    import numpy as np

    points_raw = np.arange(40).reshape((10, 4))
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




Adding Colourbars
-----------------

By default, the last column of a dataset is used to colour-code the resulting points:

::

    from pept.plots import PlotlyGrapher
    PlotlyGrapher().add_points(point_data).show()   # Colour-codes by the last column


You can change the column used to colour-code points using a numeric index (e.g. first column
``colorbar_col = 0``, second to last column ``colorbar_col = -2``) or named column (e.g.
``colorbar_col = "error"``):

::

    PlotlyGrapher().add_points(point_data, colorbar_col = -2).show()
    PlotlyGrapher().add_points(point_data, colorbar_col = "label").show()   # Coloured by trajectory
    PlotlyGrapher().add_points(point_data, colorbar_col = "v").show()       # Coloured by velocity


As a ``PlotlyGrapher`` will often manage multiple subplots, one shouldn't include explicit
colourbars on the sides *for each dataset plotted*. Therefore, colourbars are hidden by default;
add a colourbar by setting its title:

::

    PlotlyGrapher().add_points(points, colorbar_title = "Velocity").show()




Histogram of Tracking Errors
----------------------------

The ``Centroids(error = True)`` filter appends a column "error" representing the relative error
in the tracked position. You can select a named column via indexing, e.g. ``trajectories["error"]``;
you can then plot a histogram of the relative errors with:

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
    grapher.fig.write_image("figure.png", width=2560, height=1440)




Modifying the Underlying Figure
-------------------------------

You can access the Plotly figure wrapped and managed by a PlotlyGrapher using the ``.fig``
attribute:

::

    grapher.fig.update_layout(xaxis_title = "Pipe Length (mm)")


