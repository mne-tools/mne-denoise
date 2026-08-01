from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

_DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DOCS_DIR))
sys.path.insert(0, str(_DOCS_DIR.parent))
os.environ.setdefault(
    "NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache")
)
os.environ.setdefault("MNE_HOME", os.path.join(tempfile.gettempdir(), "mne_home"))

from _gallery_execution import gallery_filename_pattern  # noqa: E402

import mne_denoise  # noqa: E402

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

# Sphinx-Gallery still renders non-matching scripts as source-code pages; the
# pattern only controls whether their code is executed. Keep the full gallery
# locally by default, while allowing network-isolated builders to skip examples
# whose datasets are downloaded at runtime.
_execute_external_data_examples = os.environ.get(
    "MNE_DENOISE_DOCS_EXECUTE_EXTERNAL_DATA", "true"
).casefold() not in {"0", "false", "no"}
_gallery_filename_pattern = gallery_filename_pattern(
    execute_external_data_examples=_execute_external_data_examples
)

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": _gallery_filename_pattern,
    "ignore_pattern": r"tutorials|_legacy",
    "subsection_order": ExplicitOrder(
        [
            "../examples/dss",
            "../examples/zapline",
            "../examples/asr",
            "../examples/spectrum_interpolation",
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
