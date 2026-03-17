# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the netrl package importable from the project root
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project   = "NetRL"
copyright = "2026, Pietro Talli"
author    = "Pietro Talli"
release   = "0.2.0"
version   = "0.2"

# ---------------------------------------------------------------------------
# General Sphinx configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # API docs from docstrings
    "sphinx.ext.napoleon",       # NumPy / Google docstring style
    "sphinx.ext.viewcode",       # [source] links to highlighted code
    "sphinx.ext.autosummary",    # Summary tables for modules/classes
    "sphinx.ext.intersphinx",    # Links to external docs
    "sphinx.ext.githubpages",    # adds .nojekyll for GitHub pages
    "sphinx_design",             # grid / card / tab directives
]

templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# HTML output — ReadTheDocs theme
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "collapse_navigation": False,
}

# ---------------------------------------------------------------------------
# autodoc
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "special-members":  "__init__",
}
autodoc_member_order = "bysource"
autodoc_typehints    = "description"

# ---------------------------------------------------------------------------
# Napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring       = True
napoleon_google_docstring      = False
napoleon_include_init_with_doc = True
napoleon_use_param             = True
napoleon_use_rtype             = True

# ---------------------------------------------------------------------------
# autosummary
# ---------------------------------------------------------------------------
autosummary_generate = True

# ---------------------------------------------------------------------------
# intersphinx — cross-project links
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python":   ("https://docs.python.org/3",         None),
    "numpy":    ("https://numpy.org/doc/stable",       None),
    "gymnasium": ("https://gymnasium.farama.org",      None),
}
