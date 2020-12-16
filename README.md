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


### Example usage

A minimal analysis script using the PEPT-ML algorithm from the `pept.tracking.peptml` package:

```python
import pept
from pept.tracking import peptml

# Read in LoRs from a web-hosted CSV file.
lors_raw = pept.utilities.read_csv(
    ("https://raw.githubusercontent.com/uob-positron-imaging-centre/"
     "example_data/master/sample_2p_42rpm.csv"),    # Concatenate long string
    skiprows = 16                                   # Skip file header
)

# Encapsulate LoRs in a `LineData` subclass and compute cutpoints.
lors = pept.scanners.ParallelScreens(lors_raw, screen_separation = 712,
                                     sample_size = 200)
cutpoints = peptml.Cutpoints(lors, max_distance = 0.15)

# Cluster cutpoints using HDBSCAN and extract tracer locations.
clusterer = peptml.HDBSCANClusterer()
centres = clusterer.fit(cutpoints)

# Plot tracer locations using Plotly.
grapher = pept.visualisation.PlotlyGrapher()
grapher.add_points(centres)
grapher.show()
```

Running the above code initialises 80,000 lines of PEPT data from an online location (containing the same experiment as before - two tracers rotating at 42 RPM), transforms lines of response into accurate tracer locations and plots them in a browser-based interactive  3D graph (live version available [here](https://uob-positron-imaging-centre.github.io/live/sample_42rpm)):

![LoRs analysed using the PEPT-ML minimal script](https://github.com/uob-positron-imaging-centre/misc-hosting/blob/master/pept_centres.png?raw=true)

You can download some PEPT data samples from the [UoB Positron Imaging Centre's Repository](https://github.com/uob-positron-imaging-centre/example_data):

```
$> git clone https://github.com/uob-positron-imaging-centre/example_data
```


## Tutorials and Documentation

A very fast-paced introduction to Python is available [here](https://colab.research.google.com/drive/1Uq8Ppiv8jR-XSVsKZMcCUNuXW-l6n_RI?usp=sharing); it is aimed at engineers whose background might be a few lines written MATLAB, as well as moderate C/C++ programmers.

A beginner-friendly tutorial for using the `pept` package is available [here](https://colab.research.google.com/drive/1G8XHP9zWMMDVu23PXzANLCOKNP_RjBEO).

The links above point to Google Colaboratory, a Jupyter notebook-hosting website that lets you combine text with Python code, executing it on Google servers. Pretty neat, isn't it?

Full documentation for the `pept` package is available [here](https://uob-positron-imaging-centre.github.io).


## Performance

Significant effort has been put into making the algorithms in this package as fast as possible. The most computionally-intensive parts have been implemented in [`C`](https://github.com/uob-positron-imaging-centre/pept/search?l=c) / [`Cython`](https://github.com/uob-positron-imaging-centre/pept/search?l=Cython) and parallelised using `joblib` and `concurrent.futures.ThreadPoolExecutor`. For example, using the `peptml` subpackage, analysing 1,000,000 LoRs on the author's machine (mid 2012 MacBook Pro) takes ~26 s.

The tracking algorithms in `pept.tracking` successfully scaled up to hundreds of processors on BlueBEAR, the University of Birmingham's awesome [supercomputer](https://bear-apps.bham.ac.uk/applications/pept/0.2.2-foss-2019b-Python-3.7.4/).


## Help and Support

We recommend you check out [our tutorials](https://colab.research.google.com/drive/1G8XHP9zWMMDVu23PXzANLCOKNP_RjBEO). If your issue is not suitably resolved there, please check the [issues](https://github.com/uob-positron-imaging-centre/pept/issues) page on our GitHub. Finally, if no solution is available there, feel free to [open an issue](https://github.com/uob-positron-imaging-centre/pept/issues/new); the authors will attempt to respond as soon as possible.


## Contributing

At the moment, the subpackages in `pept.tracking` are biased towards PEPT-ML,
as there aren't many algorithms integrated into package *yet*. New algorithms
and/or recommendations for the package are more than welcome! `pept` aims to be
a community effort, be it academic, industrial, medical, or just from PEPT
enthusiasts - so it is open for help with documentation, algorithms, utilities
or analysis scripts, tutorials, and pull requests in general! To contribute please fork the project, make your changes and submit a pull request. We will do our best to work through any issues with you and get your code merged into the main branch.


## Citing

If you used this codebase or any software making use of it in a scientific
publication, we ask you to cite the following paper:

> Nicuşan AL, Windows-Yule CR. Positron emission particle tracking using machine learning. Review of Scientific Instruments. 2020 Jan 1;91(1):013329.

> https://doi.org/10.1063/1.5129251


## Licensing

The `pept` package is GNU v3.0 licensed.
Copyright (C) 2020 Andrei Leonard Nicusan.




