PEPT-ML
=======



PEPT-ML one pass of clustering recipe
-------------------------------------

::

    import pept
    from pept.tracking import *

    max_tracers = 1

    pipeline = pept.Pipeline([
        Cutpoints(max_distance = 0.5),
        HDBSCAN(true_fraction = 0.15, max_tracers = max_tracers),
        SplitLabels() + Centroids(),
        Stack(),
    ])

    locations = pipeline.fit(lors)



PEPT-ML second pass of clustering recipe
----------------------------------------

::

    import pept
    from pept.tracking import *

    max_tracers = 1

    pipeline = pept.Pipeline([
        Stack(sample_size = 30 * max_tracers, overlap = 30 * max_tracers - 1),
        HDBSCAN(true_fraction = 0.6, max_tracers = max_tracers),
        SplitLabels() + Centroids(),
        Stack(),
    ])

    locations2 = pipeline.fit(lors)



PEPT-ML complete recipe
-----------------------

Including two passes of clustering and trajectory separation:

::

    import pept
    from pept.tracking import *

    max_tracers = 1

    pipeline = pept.Pipeline([
        Cutpoints(max_distance = 0.5),
        HDBSCAN(true_fraction = 0.15, max_tracers = max_tracers),
        SplitLabels() + Centroids(),
        Stack(sample_size = 30 * max_tracers, overlap = 30 * max_tracers - 1),
        HDBSCAN(true_fraction = 0.6, max_tracers = max_tracers),
        SplitLabels() + Centroids(),
        Segregate(window = 20 * max_tracers, cut_distance = 10),
    ])

    trajectories = pipeline.fit(lors)






