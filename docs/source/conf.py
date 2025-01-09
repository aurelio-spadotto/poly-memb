# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../mesh_manager'))  # Add the library directories
sys.path.insert(0, os.path.abspath('../../solvers'))  # Add the library directories


# -- Project information -----------------------------------------------------

project = 'poly_memb'
copyright = '2024, aurelio_spadotto'
author = 'aurelio_spadotto'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',            # Automatically documents from docstrings
    'sphinx.ext.napoleon',           # To support Google and NumPy style docstrings
    'sphinx.ext.autosummary',        # Generates summary pages for modules
    'sphinx_autodoc_typehints',      # For type hints in your Python docstrings
    'nbsphinx',                      # To render Jupyter notebooks
    'sphinx.ext.mathjax',            # For rendering math in your documentation
    'sphinxcontrib.bibtex',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Mock external imports that Sphinx might try to import
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",  # add any other external dependencies you want to mock
    "Polygon",
    "sympy"
]
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# Add your .bib file(s) here
bibtex_bibfiles = ['main.bib']  # Replace with your actual .bib file name

