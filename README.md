[![PEPT Logo](https://github.com/uob-positron-imaging-centre/misc-hosting/blob/master/logo.png?raw=true)](https://github.com/uob-positron-imaging-centre/pept)

[![PyPI version shields.io](https://img.shields.io/pypi/v/pept.svg?style=flat-square)](https://pypi.python.org/pypi/pept/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pept.svg?style=flat-square)](https://anaconda.org/conda-forge/pept)
[![Documentation Status](https://readthedocs.org/projects/pept/badge/?version=latest&style=flat-square)](https://pept.readthedocs.io/en/latest/?badge=latest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1G8XHP9zWMMDVu23PXzANLCOKNP_RjBEO)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/uob-positron-imaging-centre/pept.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/uob-positron-imaging-centre/pept/context:python)
[![Language grade: C/C++](https://img.shields.io/lgtm/grade/cpp/g/uob-positron-imaging-centre/pept.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/uob-positron-imaging-centre/pept/context:cpp)
[![Azure Status](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/pept-feedstock?branchName=master)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=10178&branchName=master)
[![PyPI download month](https://img.shields.io/pypi/dm/pept.svg?style=flat-square&label=pypi%20downloads)](https://pypi.python.org/pypi/pept/)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pept.svg?style=flat-square&label=conda%20downloads)](https://anaconda.org/conda-forge/pept)
[![License: GPL v3](https://img.shields.io/github/license/uob-positron-imaging-centre/pept?style=flat-square)](https://github.com/uob-positron-imaging-centre/pept)
[![Anaconda-Platforms](https://anaconda.org/conda-forge/pept/badges/platforms.svg?style=flat-square)](https://anaconda.org/conda-forge/pept)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pept.svg?style=flat-square)](https://pypi.python.org/pypi/pept/)
[![Anaconda-Last Updated](https://anaconda.org/conda-forge/pept/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/pept)


# The `pept` Library

A Python library that integrates all the tools necessary to perform research using Positron Emission Particle Tracking (PEPT). It includes algorithms for the location, identification and tracking of particles, in addition to tools for visualisation and analysis, and utilities allowing the realistic simulation of PEPT data.


## Positron Emission Particle Tracking
PEPT is a technique developed at the University of Birmingham which allows the non-invasive, three-dimensional tracking of one or more 'tracer' particles through particulate, fluid or multiphase systems. The technique allows particle or fluid motion to be tracked with sub-millimetre accuracy and sub-millisecond temporal resolution and, due to its use of highly-penetrating 511keV gamma rays, can be used to probe the internal dynamics of even large, dense, optically opaque systems - making it ideal for industrial as well as scientific applications.

PEPT is performed by radioactively labelling a particle with a positron-emitting radioisotope such as fluorine-18 (18F) or gallium-66 (66Ga), and using the back-to-back gamma rays produced by electron-positron annihilation events in and around the tracer to triangulate its spatial position. Each detected gamma ray represents a line of response (LoR).

![Transforming LoRs into trajectories using `pept`](https://github.com/uob-positron-imaging-centre/misc-hosting/blob/master/pept_transformation.png?raw=true)
<div style = "text-align: center"> Transforming gamma rays, or lines of response (left) into individual tracer trajectories (right) using the `pept` library. Depicted is experimental data of two tracers rotating at 42 RPM, imaged using the University of Birmingham Positron Imaging Centre's parallel screens PEPT camera. </div> 


## Getting Started

These instructions will help you get started with PEPT data analysis.


### Prerequisites

This package supports Python 3.6 and above - it is built and tested for Python 3.6, 3.7 and 3.8 on Windows, Linux and macOS (thanks to [`conda-forge`](https://conda-forge.org/), which is awesome!).

You can install it using the batteries-included [Anaconda distribution](https://www.anaconda.com/products/individual) or the bare-bones [Python interpreter](https://www.python.org/downloads/). You can also check out our Python and `pept` tutorials [here](https://github.com/uob-positron-imaging-centre/tutorials).


### Installation

The easiest and quickest installation, if you are using Anaconda:

```
conda install -c conda-forge pept
```

You can also install the latest release version of `pept` from PyPI:

```
pip install --upgrade pept
```

Or you can install the development version from the GitHub repository:

```
pip install --upgrade git+https://github.com/uob-positron-imaging-centre/pept
```


### A Minimal Example Script

A minimal analysis script using the PEPT-ML algorithm from the `pept.tracking.peptml` package, transforming captured LoRs (Lines of Response, the gamma rays emitted by the tracer) into tracer locations:

```python
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
```

Running the above code initialises 80,000 lines of PEPT data from an online location (containing the same experiment as before - two tracers rotating at 42 RPM), transforms lines of response into accurate tracer locations and plots them in a browser-based interactive 3D graph (live version available [here](https://uob-positron-imaging-centre.github.io/live/sample_42rpm)):

![LoRs analysed using the PEPT-ML minimal script](https://github.com/uob-positron-imaging-centre/misc-hosting/blob/master/pept_centres.png?raw=true)

You can download some PEPT data samples from the [UoB Positron Imaging Centre's Repository](https://github.com/uob-positron-imaging-centre/example_data):

```
$> git clone https://github.com/uob-positron-imaging-centre/example_data
```


### A Complete Example Script

A complete PEPT analysis script, tracking multiple particles using the PEPT-ML algorithm, running two passes of clustering and separating out individual tracer trajectories; finally, it creates six interactive Plotly subplots that are opened in a webpage (live graph [here](https://uob-positron-imaging-centre.github.io/live/sample_full_42rpm)):

```python
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
```

The output graph is available [online here](https://uob-positron-imaging-centre.github.io/live/sample_full_42rpm).


## Tutorials and Documentation

A very fast-paced introduction to Python is available [here](https://colab.research.google.com/drive/1Uq8Ppiv8jR-XSVsKZMcCUNuXW-l6n_RI?usp=sharing); it is aimed at engineers whose background might be a few lines written MATLAB, as well as moderate C/C++ programmers.

A beginner-friendly tutorial for using the `pept` package is available [here](https://colab.research.google.com/drive/1G8XHP9zWMMDVu23PXzANLCOKNP_RjBEO).

The links above point to Google Colaboratory, a Jupyter notebook-hosting website that lets you combine text with Python code, executing it on Google servers. Pretty neat, isn't it?

Full documentation for the `pept` package is available [here](https://pept.readthedocs.io/en/latest/).


## Performance

Significant effort has been put into making the algorithms in this package as fast as possible. The most computionally-intensive parts have been implemented in [`C`](https://github.com/uob-positron-imaging-centre/pept/search?l=c) / [`Cython`](https://github.com/uob-positron-imaging-centre/pept/search?l=Cython) and parallelised using `joblib` and `concurrent.futures.ThreadPoolExecutor`. For example, using the `peptml` subpackage, analysing 1,000,000 LoRs on the author's machine (mid 2012 MacBook Pro) takes about 26 s.

The tracking algorithms in `pept.tracking` successfully scaled up to hundreds of processors on BlueBEAR, the University of Birmingham's awesome [supercomputer](https://bear-apps.bham.ac.uk/applications/pept/0.2.2-foss-2019b-Python-3.7.4/).


## Help and Support

We recommend you check out [our tutorials](https://colab.research.google.com/drive/1G8XHP9zWMMDVu23PXzANLCOKNP_RjBEO). If your issue is not suitably resolved there, please check the [issues](https://github.com/uob-positron-imaging-centre/pept/issues) page on our GitHub. Finally, if no solution is available there, feel free to [open an issue](https://github.com/uob-positron-imaging-centre/pept/issues/new); the authors will attempt to respond as soon as possible.


## Contributing

The `pept` library is not a one-man project; it is being built, improved and extended continuously (directly or indirectly) by an international team of researchers of diverse backgrounds - including programmers, mathematicians and chemical / mechanical / nuclear engineers. Want to contribute and become a PEPTspert yourself? Great, join the team!

There are multiple ways to help:
- [Open an issue](https://github.com/uob-positron-imaging-centre/pept/issues/new) mentioning any improvement you think `pept` could benefit from.
- Write a tutorial or share scripts you've developed that we can add to the [`pept` documentation](https://pept.readthedocs.io/en/latest/) to help other people in the future.
- Share your PEPT-related algorithms - tracking, post-processing, visualisation, anything really! - so everybody can benefit from them.

Want to be a superhero and contribute code directly to the library itself? Grand - fork the project, add your code and submit a pull request (if that sounds like gibberish but you're an eager programmer, check [this article](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests)). We are more than happy to work with you on integrating your code into the library and, if helpful, we can schedule a screen-to-screen meeting for a more in-depth discussion about the `pept` package architecture.

Naturally, anything you contribute to the library will respect your authorship - protected by the strong GPL v3.0 open-source license (see the "Licensing" section below). If you include published work, please add a pointer to your publication in the code documentation.


## Citing

If you used this codebase or any software making use of it in a scientific publication, we ask you to cite the following paper:

> Nicuşan AL, Windows-Yule CR. Positron emission particle tracking using machine learning. Review of Scientific Instruments. 2020 Jan 1;91(1):013329.

> https://doi.org/10.1063/1.5129251


Because `pept` is a project bringing together the expertise of many people, it hosts multiple algorithms that were developed and published in other papers. Please check the documentation of the `pept` algorithms you are using in your research and cite the original papers mentioned accordingly.


## Licensing

The `pept` package is [GPL v3.0](https://choosealicense.com/licenses/gpl-3.0/) licensed. In non-lawyer terms, the key points of this license are:
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

