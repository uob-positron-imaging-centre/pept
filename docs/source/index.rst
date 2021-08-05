..
   File   : index.rst
   License: GNU v3.0
   Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
   Date   : 28.06.2020


PEPT Documentation
==================
PEPT - a Python library that unifies Positron Emission Particle Tracking
(PEPT) research, including tracking, simulation, data analysis and
visualisation tools.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   pept_tutorials

.. toctree::
   :maxdepth: 3
   :caption: Reference:

   pept_api


Positron Emission Particle Tracking
-----------------------------------
PEPT is a technique developed at the University of Birmingham which allows the
non-invasive, three-dimensional tracking of one or more 'tracer' particles
through particulate, fluid or multiphase systems. The technique allows particle
or fluid motion to be tracked with sub-millimetre accuracy and sub-millisecond
temporal resolution and, due to its use of highly-penetrating 511keV gamma
rays, can be used to probe the internal dynamics of even large, dense,
optically opaque systems - making it ideal for industrial as well as scientific
applications [1]_.

PEPT is performed by radioactively labelling a particle with a positron-
emitting radioisotope such as fluorine-18 (18F) or gallium-68 (68Ga), and using
the back-to-back gamma rays produced by electron-positron annihilation events
in and around the tracer to triangulate its spatial position. Each detected
gamma ray represents a line of response (LoR).

.. image:: imgs/pept_transformation.png
    :alt: Transforming LoRs into trajectories using `pept`

Transforming gamma rays, or lines of response (left) into individual tracer
trajectories (right) using the `pept` library. Depicted is experimental data of
two tracers rotating at 42 RPM, imaged using the University of Birmingham
Positron Imaging Centre's parallel screens PEPT camera.


A Minimal Example Script
========================

A minimal analysis script using the PEPT-ML algorithm from the `pept.tracking.peptml` package, transforming captured LoRs (Lines of Response, the gamma rays emitted by the tracer) into tracer locations:

.. code-block:: python

    import pept
    from pept.tracking import peptml

    # Read in LoRs from a web-hosted CSV file into a NumPy array.
    lors_raw = pept.utilities.read_csv(
        "https://raw.githubusercontent.com/uob-positron-imaging-centre/example_data/master/sample_2p_42rpm.csv",
        skiprows = 16,
    )

    # Transform LoRs into the general `pept.LineData` format using a `pept.scanners`
    # converter and define a `sample_size` - i.e. the number of LoRs used to compute
    # the tracer locations in a single frame.
    lors = pept.scanners.ParallelScreens(
      lors_raw,
      screen_separation = 712,
        sample_size = 200,
    )

    # Transform LoRs into cutpoints
    cutpoints = peptml.Cutpoints(lors, max_distance = 0.15)

    # Cluster cutpoints using HDBSCAN and extract tracer locations.
    clusterer = peptml.HDBSCANClusterer()
    centres = clusterer.fit(cutpoints)

    # Plot tracer locations using Plotly.
    grapher = pept.visualisation.PlotlyGrapher()
    grapher.add_points(centres)
    grapher.show()


Running the above code initialises 80,000 lines of PEPT data from an online
location (containing the same experiment as before - two tracers rotating at 42
RPM), transforms lines of response into accurate tracer locations and plots
them in a 3D interactive browser-based graph:

.. image:: imgs/pept_centres.png
   :alt: LoRs analysed using the PEPT-ML algorithm.

You can download some PEPT data samples from the UoB Positron Imaging Centre's
repository_:

.. _repository: https://github.com/uob-positron-imaging-centre/example_data

::

    $> git clone https://github.com/uob-positron-imaging-centre/example_data


A Complete Example Script
=========================

A complete PEPT analysis script, tracking multiple particles using the PEPT-ML algorithm, running two passes of clustering and separating out individual tracer trajectories; finally, it creates six interactive Plotly subplots that are opened in a webpage (`live graph here
<https://uob-positron-imaging-centre.github.io/live/sample_full_42rpm>`_):


.. code-block::  python

    import pept
    from pept.tracking import peptml
    import pept.tracking.trajectory_separation as tsp

    from pept.visualisation import PlotlyGrapher


    # Maximum number of tracers visible at any one point
    max_tracers = 2

    # Read in LoRs from a web-hosted CSV file into a NumPy array
    lors_raw = pept.utilities.read_csv(
        "https://raw.githubusercontent.com/uob-positron-imaging-centre/example_data/master/sample_2p_42rpm.csv",
        skiprows = 16,
    )

    # 1. Transform LoRs into the general `pept.LineData` format using a `pept.scanners` converter and
    #    set a `sample_size` - i.e. the number of LoRs used to compute the tracer locations in one frame
    lors = pept.scanners.ParallelScreens(
      lors_raw,
      screen_separation = 712,
        sample_size = 200 * max_tracers,
        overlap = 100 * max_tracers,
    )

    # 2. Transform LoRs into cutpoints
    cutpoints = peptml.Cutpoints(lors, max_distance = 0.15)

    # 3. Cluster cutpoints to find particle locations (first pass of clustering)
    clusterer = peptml.HDBSCANClusterer(
        0.15 * cutpoints.sample_size / max_tracers,
        select_exemplars = True,
    )

    # Optionally find the best HDBSCAN settings for a given dataset using evolutionary optimisation
    # clusterer.optimise(cutpoints)

    centres, clustered_cutpoints = clusterer.fit(cutpoints, get_labels = True)

    # 4. Apply second pass of clustering to "tighten" trajectories
    centres.sample_size = 30 * max_tracers
    centres.overlap = centres.sample_size - 1

    clusterer2 = peptml.HDBSCANClusterer(
        0.7 * centres.sample_size / max_tracers,
        select_exemplars = True,
    )

    # Optionally find the best HDBSCAN settings for a given dataset using evolutionary optimisation
    # clusterer2.optimise(centres)

    centres2 = clusterer2.fit(centres)

    # 5. Separate out trajectories from the points found
    points_window = 20 * max_tracers
    trajectory_cut_distance = 10

    trajectories = tsp.segregate_trajectories(
        centres2,
        points_window,
        trajectory_cut_distance,
    )

    # 6. Plotting time!
    grapher = PlotlyGrapher(rows = 2, cols = 3, subplot_titles = [
        "First sample of LoRs",
        "First sample of cutpoints",
        "First sample of clustered cutpoints",
        "First pass of clustering",
        "Second pass of clustering",
        "Segregated trajectories",
    ])

    # Plot the first sample of lines and cutpoints
    grapher.add_lines(lors[0])
    grapher.add_points(cutpoints[0], col = 2)

    grapher.add_points(clustered_cutpoints[0], col = 3)
    grapher.add_points(centres, row = 2)

    grapher.add_points(centres2, row = 2, col = 2, colorbar_col = -2)
    grapher.add_points(trajectories, row = 2, col = 3)

    grapher.show()


The output graph is available online `live graph here`_.


Tutorials and Documentation
---------------------------

A very fast-paced introduction to Python is available `here (Google Colab tutorial link)
<https://colab.research.google.com/drive/1Uq8Ppiv8jR-XSVsKZMcCUNuXW-l6n_RI?usp=sharing>`_; it is aimed at engineers whose background might be a few lines written MATLAB, as well as moderate C/C++ programmers.

A beginner-friendly tutorial for using the `pept` package is available `here (Google Colab link)
<https://colab.research.google.com/drive/1G8XHP9zWMMDVu23PXzANLCOKNP_RjBEO>`_.

The links above point to Google Colaboratory, a Jupyter notebook-hosting website that lets you combine text with Python code, executing it on Google servers. Pretty neat, isn't it?

Full documentation for the `pept` package is available `here (documentation link)
<https://pept.readthedocs.io/en/latest/>`_.


Performance
-----------
Significant effort has been put into making the algorithms in this package as
fast as possible. The most compute-intensive parts have been implemented in
`C` / `Cython` and parallelised, where possible, using `joblib` and
`concurrent.futures.ThreadPoolExecutor`. For example, using the `peptml`
subpackage [2]_, analysing 1,000,000 LoRs on the author's machine (mid 2012
MacBook Pro) takes ~26 seconds.


Contributing
------------

The `pept` library is not a one-man project; it is being built, improved and extended continuously (directly or indirectly) by an international team of researchers of diverse backgrounds - including programmers, mathematicians and chemical / mechanical / nuclear engineers. Want to contribute and become a PEPTspert yourself? Great, join the team!

There are multiple ways to help:

- Open an issue mentioning any improvement you think `pept` could benefit from.
- Write a tutorial or share scripts you've developed that we can add to the `pept` documentation to help other people in the future.
- Share your PEPT-related algorithms - tracking, post-processing, visualisation, anything really! - so everybody can benefit from them.

Want to be a superhero and contribute code directly to the library itself? Grand - fork the project, add your code and submit a pull request (if that sounds like gibberish but you're an eager programmer, check `this article
<https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests>`_). We are more than happy to work with you on integrating your code into the library and, if helpful, we can schedule a screen-to-screen meeting for a more in-depth discussion about the `pept` package architecture.

Naturally, anything you contribute to the library will respect your authorship - protected by the strong GPL v3.0 open-source license (see the "Licensing" section below). If you include published work, please add a pointer to your publication in the code documentation.


Citing
------

If you used this codebase or any software making use of it in a scientific publication, we ask you to cite the following paper:

    Nicuşan AL, Windows-Yule CR. Positron emission particle tracking using machine learning. Review of Scientific Instruments. 2020 Jan 1;91(1):013329.
    https://doi.org/10.1063/1.5129251


Because `pept` is a project bringing together the expertise of many people, it hosts multiple algorithms that were developed and published in other papers. Please check the documentation of the `pept` algorithms you are using in your research and cite the original papers mentioned accordingly.


Licensing
---------

The `pept` package is `GPL v3.0
<https://choosealicense.com/licenses/gpl-3.0/>`_ licensed. In non-lawyer terms, the key points of this license are:

- You can view, use, copy and modify this code **_freely_**.
- Your modifications must _also_ be licensed with GPL v3.0 or later.
- If you share your modifications with someone, you have to include the source code as well.

Essentially do whatever you want with the code, but don't try selling it saying it's yours :). This is a community-driven project building upon many other wonderful open-source projects (NumPy, Plotly, even Python itself!) without which `pept` simply would not have been possible. GPL v3.0 is indeed a very strong *copyleft* license; it was deliberately chosen to maintain the openness and transparency of great software and progress, and respect the researchers pushing PEPT forward. Frankly, open collaboration is way more efficient than closed, for-profit competition.

Copyright (C) 2021 the `pept` developers. Until now, this library was built directly or indirectly through the brain-time of:

- Andrei Leonard Nicusan (University of Birmingham)
- Dr. Kit Windows-Yule (University of Birmingham)
- Dr. Sam Manger (University of Birmingham)
- Matthew Herald (University of Birmingham)
- Chris Jones (University of Birmingham)
- Prof. David Parker (University of Birmingham)
- Dr. Antoine Renaud (University of Edinburgh)
- Dr. Cody Wiggins (Virginia Commonwealth University)


References
----------
Papers presenting PEPT algorithms included in this library: [1]_, [2]_, [3]_.

.. [1] Parker DJ, Broadbent CJ, Fowles P, Hawkesworth MR, McNeil P. Positron
   emission particle tracking-a technique for studying flow within engineering
   equipment. Nuclear Instruments and Methods in Physics Research Section A:
   Accelerators, Spectrometers, Detectors and Associated Equipment. 1993
   Mar 10;326(3):592-607.
.. [2] Nicuşan AL, Windows-Yule CR. Positron emission particle tracking using
   machine learning. Review of Scientific Instruments. 2020 Jan 1;91(1):013329.
.. [3] Wiggins C, Santos R, Ruggles A. A feature point identification method
   for positron emission particle tracking with multiple tracers. Nuclear
   Instruments and Methods in Physics Research Section A: Accelerators,
   Spectrometers, Detectors and Associated Equipment. 2017 Jan 21;843:22-8.




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
