Interpolating Timesteps
=======================

When extracting post-processed data from tracer trajectories for e.g. probability distributions, it is often important to **sample data at fixed timesteps**. As PEPT is natively a Lagrangian technique where tracers can be tracked more often in more sensitive areas of the gamma scanners, we have to convert those "randomly-sampled" positions into regular timesteps using ``Interpolate``.

First, ``Segregate`` points into individual, continuous trajectory segments, ``GroupBy`` according to each trajectory's label, then ``Interpolate`` into regular timesteps and finally ``Stack`` them back into a ``PointData``:

::

    from pept.tracking import *

    pipe = pept.Pipeline([
        Segregate(window = 20, cut_distance = 10.),
        GroupBy("label"),
        Interpolate(timestep = 5.),
        Stack(),
    ])

    trajectories = pipe.fit(trajectories)

