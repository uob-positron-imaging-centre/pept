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


SamplesCondition
----------------

While ``Condition`` is applied on individual points, we could filter entire samples - for example, select only trajectories with more than 30 points:

::

    import pept.tracking as pt

    long_trajectories_filter = pept.Pipeline([
        # Segregate points - appends "label" column
        pt.Segregate(window = 20, cut_distance = 10),

        # Group points into samples; e.g. sample 1 contains all points with label 1
        pt.GroupBy("label"),

        # Now each sample is an entire trajectory which we can filter
        pt.SamplesCondition("sample_size > 30"),

        # And stack all remaining samples back into a single PointData
        pt.Stack(),
    ])

    long_trajectories = long_trajectories_filter.fit(trajectories)


The condition can be based on the sample itself, e.g. keep only samples that lie completely beyond x=0:

::

    # Keep only samples for which all points' X coordinates are bigger than 0
    Condition("np.all(sample['x'] > 0)")



GroupBy
-------

Stack all samples (i.e. ``LineData`` or ``PointData``) and split them into a list according to a named / numeric column index:

::

    from pept.tracking import *

    group_list = GroupBy("label").fit(trajectories)

