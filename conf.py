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


# -- Project information -----------------------------------------------------

project = 'BrainPy Examples'
copyright = '2021, BrainExamples'
author = 'BrainPy team'

# The full version, including alpha/beta/rc tags
# release = '0.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# source_suffix = '.rst'
autosummary_generate = True
napolean_use_rtype = False


# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
# We execute all notebooks, exclude the slow ones using 'exclude_patterns'
nbsphinx_execute = 'never'


# -- Options for HTML output -------------------------------------------------
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = '.rst'

# The master toctree document.
main_doc = 'index'
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'


