Saving / Loading Data
=====================

All PEPT objects can be saved in an efficient binary format using ``pept.save`` and
``pept.load``:

::

    import pept
    import numpy as np

    # Create some dummy data
    lines_raw = np.arange(70).reshape((10, 7)
    lines = pept.LineData(lines_raw)

    # Save data
    pept.save("data.pickle", lines)

    # Load data
    lines_loaded = pept.load("data.pickle")


The binary approach has the advantage of preserving all your metadata saved in the object
instances - e.g. ``columns``, ``sample_size`` - allowing the full state to be reloaded.


Matrix-like data like ``pept.LineData`` and ``pept.PointData`` can also be saved in a slower,
but human-readable CSV format using their class methods ``.to_csv``; such tabular data can then
be reinitialised using ``pept.read_csv``:

::

    # Save data in CSV format
    lines.to_csv("data.csv")

    # Load data back - *this will be a simple NumPy array!*
    lines_raw = pept.read_csv("data.csv")

    # Need to put the array back into a `pept.LineData`
    lines = pept.LineData(lines_raw)


