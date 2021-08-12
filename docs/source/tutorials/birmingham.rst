The Birmingham Method
=====================

Birmingham method recipe:

::

    import pept
    from pept.tracking import *

    pipeline = pept.Pipeline([
        BirminghamMethod(fopt = 0.5),
        Stack(),
    ])

    locations = pipeline.fit(lors)



Recipe with Trajectory Separation
---------------------------------

::

    import pept
    from pept.tracking import *

    pipeline = pept.Pipeline([
        BirminghamMethod(fopt = 0.5),
        Segregate(window = 20, cut_distance = 10),
    ])

    locations = pipeline.fit(lors)


