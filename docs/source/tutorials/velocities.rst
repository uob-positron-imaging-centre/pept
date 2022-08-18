Extracting Velocities
=====================

When extracting post-processed data from tracer trajectories for e.g. probability distributions, it is often important to **sample data at fixed timesteps**. As PEPT is natively a Lagrangian technique where tracers can be tracked more often in more sensitive areas of the gamma scanners, we have to convert those "randomly-sampled" positions into regular timesteps using ``Interpolate``.

First, ``Segregate`` points into individual, continuous trajectory segments, ``GroupBy`` according to each trajectory's label, then ``Interpolate`` into regular timesteps, then compute each point's ``Velocity`` (dimension-wise or absolute) and finally ``Stack`` them back into a ``PointData``:

::

    from pept.tracking import *

    pipe_vel = pept.Pipeline([
        Segregate(window = 20, cut_distance = 10.),
        GroupBy("label"),
        Interpolate(timestep = 5.),
        Velocity(window = 7),
        Stack(),
    ])

    trajectories = pipe_vel.fit(trajectories)


The ``Velocity`` step appends columns ``["vx", "vy", "vz"]`` (default) or ``["v"]`` (if ``absolute = True``). You can add both if you wish:

::

    from pept.tracking import *

    pept.Pipeline([
        Segregate(window = 20, cut_distance = 10.),
        GroupBy("label"),
        Interpolate(timestep = 5.),
        Velocity(window = 7),                       # Appends vx, vy, vz
        Velocity(window = 7, absolute = True),      # Appends v
        Stack(),
    ])


