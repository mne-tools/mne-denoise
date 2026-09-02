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
copyright = f"{datetime.now():%Y}, mne-denoise developers"
version = mne_denoise.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store", "changes"]
suppress_warnings = [
    "config.cache"
]  # silence sphinx-gallery "unpickleable configuration" warning

autosummary_generate = True
numpydoc_show_class_members = False

# Scientific references are kept in one BibTeX database so that method pages
# and API documentation cannot silently drift apart.
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# MyST configuration
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "ignore_pattern": r"_legacy",
    "subsection_order": ExplicitOrder(
        [
            "../examples/asr",
            "../examples/sns",
            "../examples/sound",
            "../examples/spectrum_interpolation",
            "../examples/zapline",
            "../examples/dss",
            "../examples/ssa",
            "../examples/bss_cca",
            "../examples/icanclean",
            "../examples/sspsir",
            "*",
        ]
    ),
    "within_subsection_order": FileNameSortKey,
    "reference_url": {
        "mne_denoise": None,
    },
    "download_all_examples": False,
    "show_signature": False,
    "thumbnail_size": (320, 224),
    "remove_config_comments": True,
    "min_reported_time": 1.0,
    "line_numbers": False,
    "nested_sections": True,
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
html_css_files = ["style.css"]
html_show_sourcelink = False
html_copy_source = False
html_show_sphinx = False
switcher_version_match = "dev" if ".dev" in release else version
html_theme_options = {
    "logo": {
        "text": "mne-denoise",
        "image_light": "_static/mne.svg",
        "image_dark": "_static/mne.svg",
    },
    "use_edit_page_button": switcher_version_match == "dev",
    "switcher": {
        "json_url": "https://mne.tools/mne-denoise/dev/_static/versions.json",
        "version_match": switcher_version_match,
    },
    "show_version_warning_banner": True,
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 5,
    "article_header_start": [],
    "back_to_top_button": False,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "icon_links": [
        {
            "name": "MNE Forum",
            "url": "https://mne.discourse.group/",
            "icon": "fa-solid fa-comments",
            "type": "fontawesome",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/mne-tools/mne-denoise",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
}
html_sidebars = {"index": []}
html_context = {
    "github_user": "mne-tools",
    "github_repo": "mne-denoise",
    "github_version": "main",
    "doc_path": "docs",
}
