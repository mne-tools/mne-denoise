from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime

from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

sys.path.insert(0, os.path.abspath(".."))
os.environ.setdefault(
    "NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache")
)
os.environ.setdefault("MNE_HOME", os.path.join(tempfile.gettempdir(), "mne_home"))

import mne_denoise

# -- General configuration ------------------------------------------------

project = "mne-denoise"
author = "mne-denoise developers"
copyright = f"{datetime.now():%Y}, {author}"
version = mne_denoise.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store", "changes"]
suppress_warnings = [
    "config.cache"
]  # silence sphinx-gallery "unpickleable configuration" warning

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
numpydoc_show_class_members = False

# MyST configuration
myst_heading_anchors = 3

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "ignore_pattern": r"tutorials|_legacy",
    "subsection_order": ExplicitOrder(
        [
            "../examples/sns",
            "../examples/bss_cca",
            "../examples/ssa",
            "../examples/dss",
            "../examples/zapline",
            "../examples/asr",
            "../examples/spectrum_interpolation",
            "../examples/sound",
            "../examples/sspsir",
            # Keep new example sections buildable until they receive an
            # intentional position above this fallback.
            "*",
        ]
    ),
    "within_subsection_order": FileNameSortKey,
    "reference_url": {
        "mne_denoise": None,
    },
    "download_all_examples": False,
    "show_signature": False,
    "min_reported_time": 0,
    "plot_gallery": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
}

# -- HTML -----------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/mne-tools/mne-denoise",
    "use_edit_page_button": True,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_context = {
    "github_user": "mne-tools",
    "github_repo": "mne-denoise",
    "github_version": "main",
    "doc_path": "docs",
}
