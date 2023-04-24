Trajectory Separation
=====================


Segregate Points
----------------

We can separate out trajectory segments / points that are spatio-temporally far away to:

1. Remove spurious, noisy points.
2. Separate out continuous trajectory segments.

The *spatio-temporal metric* differentiates between points that may be in the same location at different times. This is achieved by allowing points to be connected in a sliding window approach.

The ``pept.tracking.Segregate`` algorithm works by creating a *Minimum Spanning Tree* (MST, or minimum distance path) connecting all points in a dataset, then *cutting* all paths longer than a ``cut_distance``. All distinct segments are assigned a trajectory ``'label'`` (integer starting from 0); trajectories with fewer than ``min_trajectory_size`` points are considered noise (label `-1`).


::

    from pept.tracking import *

    trajectories = Segregate(window = 20, cut_distance = 10.).fit(trajectories)


Consider all trajectories with fewer than 50 points to be noise:


::

    segr = Segregate(
        window = 20,
        cut_distance = 10.,
        min_trajectory_size = 50,
    )

    trajectories = segr.fit(trajectories)


This step adds a new column "label". We can group each individual trajectory into a list with ``GroupBy``:

::

    traj_list = GroupBy("label").fit(trajectories)
    traj_list[0]    # First trajectory


*[New in pept-0.5.2]* Only connect points within a time interval; in other words, disconnect into different trajectories points whose timestamps are further apart than ``max_time_interval``:

::

     segr = Segregate(
        window = 20,
        cut_distance = 10.,
        min_trajectory_size = 50,
        max_time_interval = 2000,       # Disconnect tracer with >2s gap
    )

    trajectories = segr.fit(trajectories)
   
