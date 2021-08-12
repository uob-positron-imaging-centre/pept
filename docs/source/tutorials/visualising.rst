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






