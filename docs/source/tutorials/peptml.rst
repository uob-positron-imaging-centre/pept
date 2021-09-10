PEPT-ML
=======

PEPT using Machine Learning is a modern clustering-based tracking method that was developed specifically for noisy, fast applications.

If you are using PEPT-ML in your research, you are kindly asked to cite the following paper:

    *Nicuşan AL, Windows-Yule CR. Positron emission particle tracking using machine learning. Review of Scientific Instruments. 2020 Jan 1;91(1):013329.*


PEPT-ML one pass of clustering recipe
-------------------------------------

The LoRs are first converted into ``Cutpoints``, which are then assigned cluster labels using ``HDBSCAN``; the cutpoints are then grouped into clusters using ``SplitLabels`` and the clusters' ``Centroids`` are taken as the particle locations. Finally, stack all centroids into a single ``PointData``.

::

    import pept
    from pept.tracking import *

    max_tracers = 1

    pipeline = pept.Pipeline([
        Cutpoints(max_distance = 0.5),
        HDBSCAN(true_fraction = 0.2, max_tracers = max_tracers),
        SplitLabels() + Centroids(),
        Stack(),
    ])

    locations = pipeline.fit(lors)



PEPT-ML second pass of clustering recipe
----------------------------------------

The particle locations will always have a bit of *scatter* to them; we can *tighten* those points into accurate, dense trajectories using a *second pass of clustering*.

Set a very small sample size and maximum overlap to minimise temporal smoothing effects, then recluster the tracer locations, split according to cluster label, compute centroids, and stack into a final ``PointData``.


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
Including an example ADAC Forte data initisalisation, two passes of clustering,
trajectory separation, plotting and saving trajectories as CSV.


::

    # Import what we need from the `pept` library
    import pept
    from pept.tracking import *
    from pept.plots import PlotlyGrapher, PlotlyGrapher2D


    # Open interactive plots in the web browser
    import plotly
    plotly.io.renderers.default = "browser"


    # Initialise data from file and set sample size and overlap
    filepath = "DS1.da01"
    max_tracers = 1

    lors = pept.scanners.adac_forte(
        filepath,
        sample_size = 200 * max_tracers,
        overlap = 150 * max_tracers,
    )


    # Select only the first 1000 samples of LoRs for testing; comment out for all
    lors = lors[:1000]


    # Create PEPT-ML processing pipeline
    pipeline = pept.Pipeline([

        # First pass of clustering
        Cutpoints(max_distance = 0.2),
        HDBSCAN(true_fraction = 0.2, max_tracers = max_tracers),
        SplitLabels() + Centroids(),

        # Second pass of clustering
        Stack(sample_size = 30 * max_tracers, overlap = 30 * max_tracers - 1),
        HDBSCAN(true_fraction = 0.6, max_tracers = max_tracers),
        SplitLabels() + Centroids(),

        # Trajectory separation
        Segregate(window = 20 * max_tracers, cut_distance = 10),
        Stack(),
    ])


    # Process all samples in `lors` in parallel, using `max_workers` threads
    trajectories = pipeline.fit(lors, max_workers = 16)


    # Save trajectories as CSV
    trajectories.to_csv(filepath + ".csv")


    # Plot trajectories - first a 2D timeseries, then all 3D positions
    PlotlyGrapher2D().add_timeseries(trajectories).show()
    PlotlyGrapher().add_points(trajectories).show()


