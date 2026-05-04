# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'torch-openreml'
copyright = '2026, Patrick Li'
author = 'Patrick Li'
release = '0.1.0-alpha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon', "sphinx.ext.autosummary", "sphinx.ext.mathjax", "jupyter_sphinx"]

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'special-members': "__call__",
    'exclude-members': "__init__",
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
    'inherited-members': False
}

autosummary_generate = True
autodoc_member_order = 'bysource'
autoclass_content = 'both'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

package_path = os.path.abspath('../..')
os.environ['PYTHONPATH'] = ':'.join((package_path, os.environ.get('PYTHONPATH', '')))