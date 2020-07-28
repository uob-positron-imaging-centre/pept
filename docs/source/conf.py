#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : conf.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 29.06.2020


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


import os
import sys

sys.path.insert(0, os.path.abspath('../../pept'))


# -- Project information -----------------------------------------------------

project = 'pept'
copyright = '2020, University of Birmingham Positron Imaging Centre'
author = 'University of Birmingham Positron Imaging Centre'

# The full version, including alpha/beta/rc tags
# Load the package's __version__.py module as a dictionary.
about = {}
with open('../../pept/__version__.py') as f:
    exec(f.read(), about)

release = about["__version__"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',                 # Read the Docs theme
    'sphinx.ext.autodoc',               # Generate API from docstrings
    'numpydoc',                         # Use NumPy style docstrings

    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'autodocsumm',                      # Include autodoc modules in ToC

    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
]

# Master (or index) document
master_doc = 'index'

# Inheritance diagrams
inheritance_graph_attrs = dict(rankdir="TB", size='""')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


