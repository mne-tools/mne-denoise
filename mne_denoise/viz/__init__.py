"""Visualization functions for MNE-Denoise."""

from ._theme import (
    COLORS,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    FONTS,
    pub_figure,
    pub_legend,
    set_pub_style,
    style_axes,
)
from .benchmark import (
    DEFAULT_METHOD_COLORS,
    DEFAULT_METHOD_LABELS,
    DEFAULT_METHOD_ORDER,
    plot_harmonic_attenuation,
    plot_metric_bars,
    plot_paired_metrics,
    plot_psd_gallery,
    plot_qc_psd,
    plot_r_comparison,
    plot_subject_psd_overlay,
    plot_tradeoff_and_r,
    plot_tradeoff_scatter,
)
from .comparison import (
    plot_denoising_summary,
    plot_evoked_comparison,
    plot_overlay_comparison,
    plot_power_map,
    plot_psd_comparison,
    plot_spectral_psd_comparison,
    plot_spectrogram_comparison,
    plot_time_course_comparison,
)
from .components import (
    plot_component_image,
    plot_component_spectrogram,
    plot_component_summary,
    plot_component_time_series,
    plot_narrowband_scan,
    plot_score_curve,
    plot_spatial_patterns,
    plot_tf_mask,
)
from .dss import (
    plot_dss_comparison,
    plot_dss_eigenvalues,
    plot_dss_patterns,
    plot_dss_segmented_summary,
    plot_dss_summary,
)
from .zapline import (
    plot_adaptive_summary,
    plot_cleaning_summary,
    plot_component_scores,
    plot_zapline_analytics,
    plot_zapline_summary,
)
from .zapline import plot_psd_comparison as plot_zapline_psd_comparison
from .zapline import (
    plot_spatial_patterns as plot_zapline_patterns,
)

__all__ = [
    # Theme
    "COLORS",
    "FONTS",
    "DEFAULT_FIGSIZE",
    "DEFAULT_DPI",
    "set_pub_style",
    "style_axes",
    "pub_figure",
    "pub_legend",
    # Component plots
    "plot_component_summary",
    "plot_spatial_patterns",
    "plot_score_curve",
    "plot_component_image",
    "plot_component_time_series",
    "plot_narrowband_scan",
    "plot_tf_mask",
    "plot_component_spectrogram",
    # Comparison plots
    "plot_psd_comparison",
    "plot_time_course_comparison",
    "plot_evoked_comparison",
    "plot_spectrogram_comparison",
    "plot_power_map",
    "plot_denoising_summary",
    "plot_spectral_psd_comparison",
    "plot_overlay_comparison",
    # ZapLine plots
    "plot_zapline_analytics",
    "plot_zapline_summary",
    "plot_adaptive_summary",
    "plot_cleaning_summary",
    "plot_component_scores",
    "plot_zapline_patterns",
    "plot_zapline_psd_comparison",
    # Generalized DSS plots
    "plot_dss_summary",
    "plot_dss_segmented_summary",
    "plot_dss_comparison",
    "plot_dss_eigenvalues",
    "plot_dss_patterns",
    # Benchmark plots
    "DEFAULT_METHOD_COLORS",
    "DEFAULT_METHOD_LABELS",
    "DEFAULT_METHOD_ORDER",
    "plot_psd_gallery",
    "plot_subject_psd_overlay",
    "plot_metric_bars",
    "plot_tradeoff_scatter",
    "plot_r_comparison",
    "plot_harmonic_attenuation",
    "plot_paired_metrics",
    "plot_qc_psd",
    "plot_tradeoff_and_r",
]
