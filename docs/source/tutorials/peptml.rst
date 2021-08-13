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
        Cutpoints(max_distance = 0.5),
        HDBSCAN(true_fraction = 0.15, max_tracers = max_tracers),
        SplitLabels() + Centroids(),

        # Second pass of clustering
        Stack(sample_size = 30 * max_tracers, overlap = 30 * max_tracers - 1),
        HDBSCAN(true_fraction = 0.6, max_tracers = max_tracers),
        SplitLabels() + Centroids(),

        # Trajectory separation
        Segregate(window = 20 * max_tracers, cut_distance = 10),
    ])


    # Process all samples in `lors` in parallel, using `max_workers` threads
    trajectories = pipeline.fit(lors, max_workers = 16)


    # Save trajectories as CSV
    trajectories.to_csv(filepath + ".csv")


    # Plot trajectories - first a 2D timeseries, then all 3D positions
    PlotlyGrapher2D().add_timeseries(trajectories).show()
    PlotlyGrapher().add_points(trajectories).show()


