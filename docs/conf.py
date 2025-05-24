# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Dataset Library for 3D Machine Learning'
copyright = '2025, Eya Chouaib, Firas Drass, \u200bJana Huhne, Ole Petersen,\u200b Beatrice Picco, Daniel Schenk\u200b'
author = 'Eya Chouaib, Firas Drass, \u200bJana Huhne, Ole Petersen,\u200b Beatrice Picco, Daniel Schenk\u200b'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
