Filtering Data
==============

There are many filters in ``pept.tracking``, you can check out the Manual at the top of the page for a complete list. Here are examples with the most important ones.


Remove
------

Simply remove a column:

::

    from pept.tracking import *

    trajectories = Remove("label").fit(trajectories)


Or multiple columns:

::

    trajectories = Remove("label", "error").fit(trajectories)


Condition
---------

One of the most important filters, selecting only data that satisfies a condition:

::

    from pept.tracking import *

    trajectories = Condition("error < 15").fit(trajectories)


Or multiple ones:

::

    trajectories = Condition("error < 15, label >= 0").fit(trajectories)


In the simplest case, you just use the column name **as the first argument** followed by a comparison. If the column name is not the first argument, you must use single quotes:

::

    trajectories = Condition("0 <= 'label'").fit(trajectories)


You can also use filtering functions from NumPy in the condition string (i.e. anything returning a boolean mask):

::

    # Remove all NaNs and Infs from the 'x' column
    trajectories = Condition("np.isfinite('x')")


Finally, you can supply your own function receiving a NumPy array of the data and returning a boolean mask:

::

    def last_column_filter(data):
        return data[:, -1] > 10

    trajectories = Condition(last_column_filter).fit(trajectories)


Or using inline functions (i.e. ``lambda``):

::

    # Select points within a vertical cylinder with radius 10
    trajectories = Condition(lambda x: x[:, 1]**2 + x[:, 3]**2 < 10**2).fit(trajectories)



SplitAll
--------

Stack all samples (i.e. ``LineData`` or ``PointData``) and split them into a list according to a named / numeric column index:

::

    from pept.tracking import *

    group_list = SplitAll("label").fit(trajectories)

