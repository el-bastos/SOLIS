#!/usr/bin/env python3
"""
Plot Style Configuration for SOLIS
===================================

Defines the PlotStyle dataclass that controls all visual properties of
SOLIS plots: figure dimensions, fonts, line widths, colors, grid, and
export settings.

Default values exactly match the hardcoded constants used prior to this
module, ensuring zero visual change when no customization is applied.
"""

from dataclasses import dataclass, field, fields
from typing import Optional, Tuple, Dict

from utils.logger_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Legacy figure sizes (inches) — exact values from pre-refactor code
# Used when width_mm=None and proportion='auto' to preserve pixel-identical output
# =============================================================================

LEGACY_FIGSIZES: Dict[str, Tuple[float, float]] = {
    'decay_single': (7, 8),
    'decay_mean': (7, 8),
    'decay_batch': (7, 8),
    'decay_merged': (7, 10),
    'spectrum_single': (8, 5),
    'spectrum_merged': (10, 6),
    'spectrum_overlay': (9, 5),
    'surplus': (12, 10),
    'het_fit': (6, 5.5),
    'het_grid': (7, 6),
    'het_components': (10, 6),
    'linearity': (8, 6),
}

# Default category when an unknown category is requested
_DEFAULT_FIGSIZE = (8, 6)

# =============================================================================
# Proportion presets — width:height ratios
# =============================================================================

PROPORTIONS: Dict[str, float] = {
    'landscape': 1.4,
    'standard': 1.2,
    'square': 1.0,
    'portrait': 0.85,
    'tall_portrait': 0.7,
}

# mm → inches conversion factor
_MM_TO_INCHES = 1.0 / 25.4

# Grid linestyle map
_GRID_LINESTYLE_MAP = {
    'solid': '-',
    'dashed': '--',
    'dotted': ':',
}


@dataclass
class PlotStyle:
    """
    Centralized plot styling configuration for all SOLIS plotters.

    Default values match the hardcoded constants used before this module
    existed, so PlotStyle() produces pixel-identical output.
    """

    # --- Dimensions ---
    width_mm: Optional[float] = None
    """Figure width in mm. None = Auto (use legacy per-plot-type sizes)."""

    proportion: str = 'auto'
    """Aspect ratio preset: 'auto', 'landscape', 'standard', 'square',
    'portrait', 'tall_portrait'."""

    # --- Fonts ---
    font_family: str = 'Arial'
    font_size_title: int = 14
    font_size_axis_label: int = 12
    font_size_tick_label: int = 10
    font_size_legend: int = 10
    font_size_annotation: int = 9
    font_weight_title: str = 'bold'
    """Title font weight: 'normal' or 'bold'."""

    # --- Lines ---
    linewidth_data: float = 1.5
    linewidth_fit: float = 2.0
    linewidth_residuals: float = 1.0
    markersize_data: int = 3

    # --- Grid ---
    grid_enabled: bool = True
    grid_style: str = 'dashed'
    """Grid line style: 'solid', 'dashed', or 'dotted'."""

    grid_alpha: float = 0.3

    # --- Colors ---
    color_data: str = '#646464'
    color_fit: str = '#CC0000'
    color_fit_alt: str = '#FF8C00'
    """Secondary fit color (e.g., literature model)."""

    color_residuals: str = '#CC0000'

    # --- Export ---
    export_dpi: int = 300

    # -----------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------

    def get_figsize(self, category: str = 'decay_single') -> Tuple[float, float]:
        """
        Return screen display size — always legacy per-category sizes.

        Screen display always uses legacy sizes to ensure correct on-screen
        appearance. Custom dimensions (width_mm, proportion) only apply
        at export time via ``get_export_figsize()``.

        Parameters
        ----------
        category : str
            Plot category key (e.g. 'decay_single', 'spectrum_merged').

        Returns
        -------
        tuple[float, float]
            (width_inches, height_inches)
        """
        return LEGACY_FIGSIZES.get(category, _DEFAULT_FIGSIZE)

    def get_export_figsize(self, category: str = 'decay_single') -> Optional[Tuple[float, float]]:
        """
        Return export size in inches, or None if width_mm is not set.

        When None is returned, export should use the current screen size.

        Parameters
        ----------
        category : str
            Plot category key (e.g. 'decay_single', 'spectrum_merged').

        Returns
        -------
        tuple[float, float] or None
            (width_inches, height_inches) for export, or None to export
            at screen size.
        """
        if self.width_mm is None:
            return None  # Export at screen size

        w = self.width_mm * _MM_TO_INCHES

        if self.proportion == 'auto':
            # Custom width but auto proportion: preserve legacy aspect ratio
            legacy = LEGACY_FIGSIZES.get(category, _DEFAULT_FIGSIZE)
            ratio = legacy[0] / legacy[1]
            h = w / ratio
        else:
            ratio = PROPORTIONS.get(self.proportion, 1.2)
            h = w / ratio

        return (w, h)

    def get_grid_linestyle(self) -> str:
        """Return matplotlib linestyle string for the grid."""
        return _GRID_LINESTYLE_MAP.get(self.grid_style, '--')

    def apply_to_figure(self, fig) -> None:
        """Apply font family to a matplotlib Figure via rcParams."""
        import matplotlib as mpl
        with mpl.rc_context({'font.family': self.font_family}):
            # rc_context only works as context manager, so we set directly
            pass
        # Set on the figure's own renderer
        for text in fig.texts:
            text.set_fontfamily(self.font_family)

    def configure_grid(self, ax) -> None:
        """Apply grid settings to a matplotlib Axes."""
        if self.grid_enabled:
            ax.grid(True, alpha=self.grid_alpha,
                    linestyle=self.get_grid_linestyle())
        else:
            ax.grid(False)

    # -----------------------------------------------------------------
    # Serialization (plain dict, for preferences storage)
    # -----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a plain dict (only non-default values)."""
        default = PlotStyle()
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            default_val = getattr(default, f.name)
            if val != default_val:
                result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, d: dict) -> 'PlotStyle':
        """
        Create PlotStyle from a dict. Missing keys use defaults.

        Gracefully ignores unknown keys for forward compatibility.
        """
        if not d:
            return cls()
        valid_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_names}
        return cls(**filtered)
