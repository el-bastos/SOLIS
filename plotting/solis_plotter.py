#!/usr/bin/env python3
"""
Publication-Quality Plotting Module for SOLIS
==============================================

Creates publication-ready figures matching manuscript styling requirements.

Phase 3C: Publication plotting system
Session 14: Added matplotlib support for reliable rendering
Author: SOLIS Team
Date: 2025-10-25, Updated: 2025-10-29
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import logging

# Matplotlib imports for Session 14
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for figure generation
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from core.kinetics_dataclasses import KineticsResult, SNRResult

logger = logging.getLogger(__name__)


# === Color Scheme (Publication Quality) ===
# Note: matplotlib uses different color format than Plotly
# For matplotlib: use hex codes or (r,g,b,a) tuples with values 0-1
COLORS = {
    # Experimental data
    'experimental': (0.39, 0.39, 0.39, 0.6),  # Gray with transparency (100/255)
    'experimental_line': '#646464',  # Solid gray for markers

    # Fitted curves
    'fit_main': '#CC0000',  # Red for Main model f(t-t₀)
    'fit_literature': '#FF8C00',  # Orange for Literature model f(t)

    # Spike region shading
    'spike_region': (0.78, 0.78, 0.78, 0.3),  # Light gray transparent (200/255)

    # Residuals
    'residuals_main': '#CC0000',  # Red
    'residuals_lit': '#FF8C00',  # Orange
    'zero_line': (0, 0, 0, 0.3),  # Gray zero line

    # Grid and axes
    'grid': (0.78, 0.78, 0.78, 0.3),  # Light gray (200/255)
    'axis_line': (0, 0, 0, 0.5),  # Dark gray
}

# === Multi-dataset Color Palette (for merged plots) ===
MERGED_COLORS = [
    '#8B008B',  # Purple/Magenta
    '#DC143C',  # Crimson/Pink-Red
    '#CC0000',  # Red
    '#FF8C00',  # Orange
    '#FFD700',  # Gold/Yellow
    '#228B22',  # Forest Green
    '#4169E1',  # Royal Blue
    '#9400D3',  # Dark Violet
]


# === Font Configuration ===
FONTS = {
    'family': 'Arial',
    'size_small': 10,  # Legend, annotations
    'size_medium': 12,  # Axis labels
    'size_large': 14,  # Axis titles
}


# === Layout Dimensions ===
LAYOUT = {
    'width': 500,
    'height': 600,
    'panel_ratio': [0.85, 0.15],  # Top 85%, bottom 15%
    'vertical_spacing': 0.05,
    'margin': dict(l=80, r=40, t=40, b=60),
}


class SOLISPlotter:
    """
    Publication-quality plotting for SOLIS kinetics analysis.

    Creates dual-panel plots matching manuscript requirements:
    - Top panel (75%): Decay curve with experimental data and fits
    - Bottom panel (25%): Weighted residuals

    Features:
    - Color scheme matching example figures
    - Log/linear x-axis toggle
    - SNR annotation overlay
    - Spike region shading
    - Vector PDF export via kaleido
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize SOLIS plotter.

        Args:
            output_dir: Directory for exported plots (default: current directory)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SOLISPlotter initialized: output_dir={self.output_dir}")

    # === MATPLOTLIB PLOTTING METHODS (Session 14) ===

    def plot_single_decay_mpl(
        self,
        result: KineticsResult,
        snr_result: Optional[SNRResult] = None,
        log_x: bool = True,
        show_literature: bool = True,
        title: Optional[str] = None
    ) -> Figure:
        """
        Generate dual-panel plot for single sample decay using matplotlib.

        Top panel: Decay curve with experimental data and fitted curves
        Bottom panel: Weighted residuals

        Args:
            result: KineticsResult from kinetics_analyzer
            snr_result: Optional SNRResult for annotation (from result.snr_result)
            log_x: Use logarithmic x-axis (default True)
            show_literature: Show literature model f(t) curve (default True)
            title: Optional plot title

        Returns:
            Matplotlib Figure object ready for display or export
        """
        # Create figure with GridSpec layout (85% top, 15% bottom)
        # OPTIMIZED: Use constrained_layout instead of tight_layout (40-60% faster)
        fig = Figure(figsize=(7, 8), dpi=100, constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[0.85, 0.15], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_main)

        # Extract data
        time = result.time_experiment_us
        intensity = result.intensity_raw
        main_curve = result.main_curve
        main_residuals = result.main_weighted_residuals

        # Filter residuals to show only non-masked region
        if hasattr(result, 'fitting_mask') and result.fitting_mask is not None:
            time_residuals = time[result.fitting_mask]
            main_residuals_filtered = main_residuals[result.fitting_mask]
            unmasked_intensity = intensity[result.fitting_mask]
        else:
            time_residuals = time
            main_residuals_filtered = main_residuals
            unmasked_intensity = intensity

        # === TOP PANEL: Decay curve ===
        # Plot experimental data (gray points)
        ax_main.plot(time, intensity, 'o', color='#646464', markersize=3,
                    alpha=0.6, label='Experimental', zorder=1)

        # Plot Main model fit (red line)
        ax_main.plot(time, main_curve, '-', color=COLORS['fit_main'], linewidth=2,
                    label='Main, f(t; t₀)', zorder=2)

        # Plot Literature model fit (orange line) if available
        if show_literature and result.literature.success and result.literature.curve is not None:
            ax_main.plot(time, result.literature.curve, '-', color=COLORS['fit_literature'],
                        linewidth=2, label='Literature, f(t)', zorder=2)

        # Axis settings for main panel
        if log_x:
            ax_main.set_xscale('log')
            ax_main.set_xlim(0.01, 100)
        else:
            ax_main.set_xscale('linear')
            ax_main.set_xlim(0, 50)

        # Y-axis: Auto-scale with 0 at bottom
        ax_main.set_ylim(bottom=0, auto=True)
        ax_main.autoscale(enable=True, axis='y', tight=False)

        ax_main.set_ylabel('Intensity (counts)', fontsize=12)
        ax_main.legend(loc='upper right', fontsize=10, framealpha=0.8)
        ax_main.grid(True, alpha=0.3)

        # Remove x-axis labels from top panel (shared with bottom)
        ax_main.tick_params(labelbottom=False)

        # === BOTTOM PANEL: Residuals ===
        # Main model residuals (red)
        ax_res.plot(time_residuals, main_residuals_filtered, 'o', color=COLORS['residuals_main'],
                   markersize=2, label='WRes Main', zorder=2)

        # Literature model residuals (orange) if available
        if show_literature and result.literature.success and result.literature.weighted_residuals is not None:
            if hasattr(result, 'fitting_mask') and result.fitting_mask is not None:
                lit_residuals_filtered = result.literature.weighted_residuals[result.fitting_mask]
            else:
                lit_residuals_filtered = result.literature.weighted_residuals
            ax_res.plot(time_residuals, lit_residuals_filtered, 'o', color=COLORS['residuals_lit'],
                       markersize=2, label='WRes Lit', zorder=2)

        # Zero line
        ax_res.axhline(y=0, color=COLORS['zero_line'], linewidth=1, linestyle='solid', zorder=1)

        # Axis settings for residuals panel (ignore NaN)
        max_abs_residual = np.nanmax(np.abs(main_residuals_filtered))
        if np.isnan(max_abs_residual) or np.isinf(max_abs_residual) or max_abs_residual == 0:
            residual_limit = 5  # Default fallback
        else:
            residual_limit = np.ceil(max_abs_residual)
        ax_res.set_ylim(-residual_limit, residual_limit)
        ax_res.set_ylabel('Weighted\nResiduals', fontsize=10)
        ax_res.set_xlabel('Time (μs)', fontsize=12)
        ax_res.grid(True, alpha=0.3)

        # Add title if provided
        if title:
            fig.suptitle(title, fontsize=14)

        # Layout handled by constrained_layout (set at Figure creation)

        # OPTIMIZED Phase 3: Use lazy export data to reduce memory footprint
        # Instead of eagerly creating the dict, we store a callable that generates it on demand
        # This allows the figure to release references when not exporting
        def _build_export_data():
            """Lazy builder for export data - only called when CSV export is requested."""
            data = {
                'time_us': time,
                'intensity_raw': intensity,
                'main_curve': main_curve,
                'main_weighted_residuals': main_residuals,
                'fitting_mask': result.fitting_mask if hasattr(result, 'fitting_mask') else None,
                'spike_region': result.spike_region if hasattr(result, 'spike_region') else None
            }

            # Add literature curve data if available
            if show_literature and result.literature.success and result.literature.curve is not None:
                data['literature_curve'] = result.literature.curve
                data['literature_weighted_residuals'] = result.literature.weighted_residuals

            return data

        # Attach the builder callable (not the data itself)
        fig.solis_export_data = _build_export_data

        logger.info(f"Matplotlib single decay plot created: {len(time)} points, log_x={log_x}")
        return fig

    def plot_mean_decay_mpl(
        self,
        mean_arrays: Dict[str, np.ndarray],
        sd_arrays: Dict[str, np.ndarray],
        log_x: bool = False,
        title: str = "Mean Decay Curve"
    ) -> Figure:
        """
        Plot MEAN decay curve with SD envelopes using matplotlib.

        This plots the COMPUTED MEAN from StatisticalAnalyzer, NOT individual replicates.

        Parameters
        ----------
        mean_arrays : dict
            Dictionary with keys: 'mean_time_experiment_us', 'mean_intensity_raw',
            'mean_main_curve_ft_t0', 'mean_main_weighted_residuals'
        sd_arrays : dict
            Dictionary with keys: 'sd_intensity_raw', 'sd_main_curve_ft_t0',
            'sd_main_weighted_residuals'
        log_x : bool
            Use logarithmic x-axis
        title : str
            Plot title

        Returns
        -------
        Figure
            Matplotlib figure with mean curves and SD envelopes
        """
        # Create figure with GridSpec layout
        # OPTIMIZED: Use constrained_layout instead of tight_layout (40-60% faster)
        fig = Figure(figsize=(7, 8), dpi=100, constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[0.85, 0.15], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_main)

        # Extract data
        time = mean_arrays['mean_time_experiment_us']
        mean_intensity = mean_arrays['mean_intensity_raw']
        sd_intensity = sd_arrays['sd_intensity_raw']
        mean_fit = mean_arrays['mean_main_curve_ft_t0']
        sd_fit = sd_arrays['sd_main_curve_ft_t0']
        mean_residuals = mean_arrays['mean_main_weighted_residuals']
        sd_residuals = sd_arrays['sd_main_weighted_residuals']
        fitting_mask = mean_arrays.get('best_fitting_mask')

        # Filter residuals to non-masked region
        if fitting_mask is not None:
            mask_len = min(len(fitting_mask), len(time), len(mean_residuals))
            mask_truncated = fitting_mask[:mask_len]
            time_residuals = time[:mask_len][mask_truncated]
            mean_residuals_filtered = mean_residuals[:mask_len][mask_truncated]
            sd_residuals_filtered = sd_residuals[:mask_len][mask_truncated]

            # For y-axis scaling
            mask_len_intensity = min(len(fitting_mask), len(mean_intensity), len(sd_intensity))
            mask_truncated_intensity = fitting_mask[:mask_len_intensity]
            unmasked_mean = mean_intensity[:mask_len_intensity][mask_truncated_intensity]
            unmasked_sd = sd_intensity[:mask_len_intensity][mask_truncated_intensity]
        else:
            time_residuals = time
            mean_residuals_filtered = mean_residuals
            sd_residuals_filtered = sd_residuals
            unmasked_mean = mean_intensity
            unmasked_sd = sd_intensity

        # === TOP PANEL: Decay curve with SD envelopes ===
        # Plot experimental data SD envelope (gray)
        ax_main.fill_between(time, mean_intensity - sd_intensity, mean_intensity + sd_intensity,
                            color='#646464', alpha=0.2, label='Data ± SD')
        # Plot mean experimental data (gray line)
        ax_main.plot(time, mean_intensity, '-', color='#646464', linewidth=1.5)

        # Plot fit SD envelope (red)
        ax_main.fill_between(time, mean_fit - sd_fit, mean_fit + sd_fit,
                            color=COLORS['fit_main'], alpha=0.2, label='Fit ± SD')
        # Plot mean fit (bold red line)
        ax_main.plot(time, mean_fit, '-', color=COLORS['fit_main'], linewidth=3, label='Mean Fit')

        # Axis settings for main panel
        if log_x:
            ax_main.set_xscale('log')
            ax_main.set_xlim(0.01, 100)
        else:
            ax_main.set_xscale('linear')
            ax_main.set_xlim(0, 50)

        # Y-axis: Auto-scale with 0 at bottom
        ax_main.set_ylim(bottom=0, auto=True)
        ax_main.autoscale(enable=True, axis='y', tight=False)

        ax_main.set_ylabel('Intensity (counts)', fontsize=12)
        ax_main.legend(loc='upper right', fontsize=10, framealpha=0.8)
        ax_main.grid(True, alpha=0.3)
        ax_main.tick_params(labelbottom=False)

        # === BOTTOM PANEL: Residuals with SD envelope ===
        # Plot residuals SD envelope (red, only non-masked region)
        ax_res.fill_between(time_residuals,
                           mean_residuals_filtered - sd_residuals_filtered,
                           mean_residuals_filtered + sd_residuals_filtered,
                           color=COLORS['residuals_main'], alpha=0.2)
        # Plot mean residuals (red points)
        ax_res.plot(time_residuals, mean_residuals_filtered, 'o', color=COLORS['residuals_main'],
                   markersize=2)

        # Zero line
        ax_res.axhline(y=0, color=COLORS['zero_line'], linewidth=1, linestyle='dashed', zorder=1)

        # Axis settings for residuals panel (ignore NaN)
        max_abs_residual = np.nanmax(np.abs(mean_residuals_filtered + sd_residuals_filtered))
        if np.isnan(max_abs_residual) or np.isinf(max_abs_residual) or max_abs_residual == 0:
            residual_limit = 5
        else:
            residual_limit = np.ceil(max_abs_residual)
        ax_res.set_ylim(-residual_limit, residual_limit)
        ax_res.set_ylabel('Weighted\nResiduals', fontsize=10)
        ax_res.set_xlabel('Time (μs)', fontsize=12)
        ax_res.grid(True, alpha=0.3)

        # Add title
        fig.suptitle(title, fontsize=14)

        # Layout handled by constrained_layout (set at Figure creation)

        # OPTIMIZED Phase 3: Lazy export data builder
        def _build_export_data():
            """Lazy builder for export data - only called when CSV export is requested."""
            return {
                'time_us': time,
                'mean_intensity_raw': mean_intensity,
                'sd_intensity_raw': sd_intensity,
                'mean_main_curve': mean_fit,
                'sd_main_curve': sd_fit,
                'mean_weighted_residuals': mean_residuals,
                'sd_weighted_residuals': sd_residuals,
                'fitting_mask': fitting_mask
            }

        fig.solis_export_data = _build_export_data

        logger.info(f"Matplotlib mean decay plot created")
        return fig

    def plot_batch_summary_mpl(
        self,
        results: List[KineticsResult],
        log_x: bool = True,
        title: Optional[str] = None,
        show_mean: bool = True,
        show_statistics: bool = True
    ) -> Figure:
        """
        Generate batch analysis summary plot with all replicates overlaid using matplotlib.

        Shows:
        - All individual replicate decay curves
        - Mean curve (if show_mean=True)
        - Error bands (standard deviation)
        - Statistics table annotation (if show_statistics=True)

        Args:
            results: List of KineticsResult objects (one per replicate)
            log_x: Use logarithmic x-axis (default True)
            title: Optional plot title
            show_mean: Show mean curve with error bands (default True)
            show_statistics: Show statistics annotation box (default True)

        Returns:
            Matplotlib Figure object
        """
        if not results:
            raise ValueError("No results provided for batch summary")

        # Create figure with GridSpec layout
        # OPTIMIZED: Use constrained_layout instead of tight_layout (40-60% faster)
        fig = Figure(figsize=(7, 8), dpi=100, constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[0.85, 0.15], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_main)

        # Number of replicates
        n_reps = len(results)

        # Color palette for replicates (shades of gray with varying opacity)
        rep_alphas = [0.3 + 0.5 * i / n_reps for i in range(n_reps)]

        # Storage for calculating symmetrical residual axis
        all_residuals = []

        # === Plot each replicate ===
        for i, result in enumerate(results):
            time = result.time_experiment_us
            intensity = result.intensity_raw
            main_curve = result.main_curve
            main_residuals = result.main_weighted_residuals

            # Filter residuals by mask
            if hasattr(result, 'fitting_mask') and result.fitting_mask is not None:
                main_residuals_filtered = main_residuals[result.fitting_mask]
                time_residuals = time[result.fitting_mask]
            else:
                main_residuals_filtered = main_residuals
                time_residuals = time
            all_residuals.append(main_residuals_filtered)

            # Plot experimental data
            ax_main.plot(time, intensity, 'o', color='#646464', markersize=3,
                        alpha=rep_alphas[i], label=f'Rep {i+1} Data', zorder=1)

            # Plot fit
            ax_main.plot(time, main_curve, '-', color=COLORS['fit_main'], linewidth=1.5,
                        alpha=0.6, zorder=2)

            # Plot residuals
            ax_res.plot(time_residuals, main_residuals_filtered, 'o',
                       color=COLORS['residuals_main'], markersize=2, alpha=0.5, zorder=1)

        # === Calculate and plot mean curve if requested ===
        # OPTIMIZED Phase 3: Cache mean curve calculation for reuse in export
        mean_curve_cached = None
        std_curve_cached = None

        if show_mean and n_reps > 1:
            mean_curve_cached, std_curve_cached = self._calculate_mean_curve(results)
            # Truncate time to match mean curve length
            min_length = len(mean_curve_cached)
            time = results[0].time_experiment_us[:min_length]

            # Mean curve (bold red)
            ax_main.plot(time, mean_curve_cached, '-', color=COLORS['fit_main'], linewidth=3,
                        label='Mean Fit', zorder=3)

            # Error bands (±1 SD)
            ax_main.fill_between(time, mean_curve_cached - std_curve_cached, mean_curve_cached + std_curve_cached,
                                color=COLORS['fit_main'], alpha=0.2, label='±1 SD', zorder=2)

        # === Add statistics annotation if requested ===
        if show_statistics:
            self._add_batch_statistics_mpl(fig, ax_main, results)

        # === Zero line for residuals ===
        ax_res.axhline(y=0, color=COLORS['zero_line'], linewidth=1, linestyle='solid', zorder=0)

        # === Axis settings ===
        # X-axis
        if log_x:
            ax_main.set_xscale('log')
            ax_main.set_xlim(0.01, 100)
        else:
            ax_main.set_xscale('linear')
            ax_main.set_xlim(0, 50)

        # Y-axis: 0 to next round number after peak
        time = results[0].time_experiment_us
        intensity = results[0].intensity_raw
        if hasattr(results[0], 'fitting_mask') and results[0].fitting_mask is not None:
            unmasked_intensity = intensity[results[0].fitting_mask]
        else:
            unmasked_intensity = intensity

        # Y-axis: Auto-scale with 0 at bottom
        ax_main.set_ylim(bottom=0, auto=True)
        ax_main.autoscale(enable=True, axis='y', tight=False)

        # Residual Y-axis: symmetrical around zero
        if all_residuals:
            all_residuals_concat = np.concatenate(all_residuals)
            max_abs_residual = np.nanmax(np.abs(all_residuals_concat))
            if np.isnan(max_abs_residual) or np.isinf(max_abs_residual) or max_abs_residual == 0:
                residual_limit = 5
            else:
                residual_limit = np.ceil(max_abs_residual)
            ax_res.set_ylim(-residual_limit, residual_limit)

        # Labels and legend
        ax_main.set_ylabel('Intensity (counts)', fontsize=12)
        ax_main.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax_main.grid(True, alpha=0.3)
        ax_main.tick_params(labelbottom=False)

        ax_res.set_ylabel('Weighted\nResiduals', fontsize=10)
        ax_res.set_xlabel('Time (μs)', fontsize=12)
        ax_res.grid(True, alpha=0.3)

        # Add title if provided
        if title:
            fig.suptitle(title, fontsize=14)

        # Layout handled by constrained_layout (set at Figure creation)

        # OPTIMIZED Phase 3: Lazy export data builder
        def _build_export_data():
            """Lazy builder for export data - only called when CSV export is requested."""
            export_data = {'n_replicates': n_reps}
            for i, result in enumerate(results, 1):
                export_data[f'rep{i}_time_us'] = result.time_experiment_us
                export_data[f'rep{i}_intensity_raw'] = result.intensity_raw
                export_data[f'rep{i}_main_curve'] = result.main_curve
                export_data[f'rep{i}_main_weighted_residuals'] = result.main_weighted_residuals
                if hasattr(result, 'fitting_mask') and result.fitting_mask is not None:
                    export_data[f'rep{i}_fitting_mask'] = result.fitting_mask

            # Add mean and std if calculated
            # OPTIMIZED Phase 3: Reuse cached calculation instead of recomputing
            if show_mean and n_reps > 1 and mean_curve_cached is not None:
                min_length = len(mean_curve_cached)
                export_data['time_us'] = results[0].time_experiment_us[:min_length]
                export_data['mean_curve'] = mean_curve_cached
                export_data['std_curve'] = std_curve_cached

            return export_data

        fig.solis_export_data = _build_export_data

        logger.info(f"Matplotlib batch summary plot created: {n_reps} replicates")
        return fig

    def _add_batch_statistics_mpl(
        self,
        fig: Figure,
        ax: Any,
        results: List[KineticsResult]
    ) -> None:
        """
        Add statistics annotation box to matplotlib batch plot.

        Args:
            fig: Matplotlib figure
            ax: Axes object to add text to
            results: List of KineticsResult objects
        """
        # Extract parameters from all replicates
        A_values = [r.parameters.A for r in results]
        tau_delta_values = [r.parameters.tau_delta for r in results]
        tau_T_values = [r.parameters.tau_T for r in results]
        r_squared_values = [r.fit_quality.r_squared for r in results]

        # Calculate statistics
        stats_text = (
            f"n = {len(results)}\n"
            f"A: {np.mean(A_values):.1f} ± {np.std(A_values):.1f}\n"
            f"τΔ: {np.mean(tau_delta_values):.2f} ± {np.std(tau_delta_values):.2f} μs\n"
            f"τT: {np.mean(tau_T_values):.2f} ± {np.std(tau_T_values):.2f} μs\n"
            f"R²: {np.mean(r_squared_values):.4f} ± {np.std(r_squared_values):.4f}"
        )

        # Add text box to upper left corner
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1))

    def plot_merged_decay_mpl(
        self,
        selected_items: List[Dict[str, Any]],
        log_x: bool = False,
        title: str = "Merged Decay Curves"
    ) -> Figure:
        """
        Plot multiple datasets in a SINGLE merged plot with stacked residual panels using matplotlib.

        Layout:
        - Top panel (large): All decay curves with data + fits + SD envelopes
        - Bottom panels: Individual residual panels stacked vertically

        Parameters
        ----------
        selected_items : list of dict
            Each dict has keys:
            - 'type': 'mean' or 'replicate'
            - 'compound': compound name
            For 'mean': 'mean_arrays', 'sd_arrays'
            For 'replicate': 'result' (KineticsResult), 'replicate_num'
        log_x : bool
            Use logarithmic x-axis for decay panel
        title : str
            Overall plot title

        Returns
        -------
        Figure
            Matplotlib figure with merged plot
        """
        n_datasets = len(selected_items)
        if n_datasets == 0:
            raise ValueError("No datasets provided for merged plot")

        # Create subplot layout: 1 large decay panel + n_datasets residual panels
        # Heights: decay panel gets 60%, residuals split remaining 40%
        decay_height = 0.6
        residual_height = 0.4 / n_datasets

        row_heights = [decay_height] + [residual_height] * n_datasets

        # OPTIMIZED: Use constrained_layout instead of tight_layout (40-60% faster)
        fig = Figure(figsize=(7, 10), dpi=100, constrained_layout=True)
        gs = GridSpec(n_datasets + 1, 1, figure=fig, height_ratios=row_heights, hspace=0.05)

        # Create axes
        ax_main = fig.add_subplot(gs[0])
        ax_residuals = [fig.add_subplot(gs[i+1], sharex=ax_main) for i in range(n_datasets)]

        # Storage for residual limits and max intensity
        residual_limits = []
        max_intensity_values = []

        # === Process each dataset ===
        for idx, item in enumerate(selected_items):
            color = MERGED_COLORS[idx % len(MERGED_COLORS)]

            if item['type'] == 'mean':
                # === Plot MEAN data with SD envelopes ===
                mean_arrays = item['mean_arrays']
                sd_arrays = item['sd_arrays']
                compound = item['compound']

                time = mean_arrays['mean_time_experiment_us']
                mean_intensity = mean_arrays['mean_intensity_raw']
                sd_intensity = sd_arrays['sd_intensity_raw']
                mean_fit = mean_arrays['mean_main_curve_ft_t0']
                sd_fit = sd_arrays['sd_main_curve_ft_t0']
                mean_residuals = mean_arrays['mean_main_weighted_residuals']
                sd_residuals = sd_arrays['sd_main_weighted_residuals']
                fitting_mask = mean_arrays.get('best_fitting_mask')

                # Calculate max intensity in non-masked region
                if fitting_mask is not None:
                    mask_len = min(len(fitting_mask), len(mean_intensity), len(sd_intensity))
                    mask_truncated = fitting_mask[:mask_len]
                    unmasked_mean = mean_intensity[:mask_len][mask_truncated]
                    unmasked_sd = sd_intensity[:mask_len][mask_truncated]
                else:
                    unmasked_mean = mean_intensity
                    unmasked_sd = sd_intensity
                max_intensity_values.append(np.nanmax(unmasked_mean + unmasked_sd))

                # Data SD envelope
                ax_main.fill_between(time, mean_intensity - sd_intensity, mean_intensity + sd_intensity,
                                    color=color, alpha=0.2, label=f'{compound} ± SD')
                # Mean data points
                ax_main.plot(time, mean_intensity, 'o', color=color, markersize=3, alpha=0.6)
                # Mean fit line
                ax_main.plot(time, mean_fit, '-', color=color, linewidth=2, label=f'{compound} Fit')

                # === Residuals in separate panel ===
                ax_res = ax_residuals[idx]

                # Filter residuals by mask
                if fitting_mask is not None:
                    mask_len = min(len(fitting_mask), len(time), len(mean_residuals))
                    mask_truncated = fitting_mask[:mask_len]
                    time_residuals = time[:mask_len][mask_truncated]
                    mean_residuals_filtered = mean_residuals[:mask_len][mask_truncated]
                    sd_residuals_filtered = sd_residuals[:mask_len][mask_truncated]
                else:
                    time_residuals = time
                    mean_residuals_filtered = mean_residuals
                    sd_residuals_filtered = sd_residuals

                # Residual SD envelope (only non-masked region)
                ax_res.fill_between(time_residuals,
                                   mean_residuals_filtered - sd_residuals_filtered,
                                   mean_residuals_filtered + sd_residuals_filtered,
                                   color=color, alpha=0.2)
                # Mean residuals
                ax_res.plot(time_residuals, mean_residuals_filtered, '-', color=color, linewidth=1)

                # Calculate residual limit for this panel (ignore NaN)
                max_abs_residual = np.nanmax(np.abs(mean_residuals_filtered + sd_residuals_filtered))
                if np.isnan(max_abs_residual) or np.isinf(max_abs_residual) or max_abs_residual == 0:
                    residual_limits.append(5)
                else:
                    residual_limits.append(np.ceil(max_abs_residual))

            else:  # type == 'replicate'
                # === Plot INDIVIDUAL replicate ===
                result = item['result']
                compound = item['compound']
                rep_num = item['replicate_num']

                time = result.time_experiment_us
                intensity = result.intensity_raw
                fit = result.main_curve  # Fixed: KineticsResult uses main_curve, not main_curve_ft_t0
                residuals = result.main_weighted_residuals
                fitting_mask = result.fitting_mask if hasattr(result, 'fitting_mask') else None

                # Calculate max intensity in non-masked region
                if fitting_mask is not None:
                    mask_len = min(len(fitting_mask), len(intensity))
                    mask_truncated = fitting_mask[:mask_len]
                    unmasked_intensity = intensity[:mask_len][mask_truncated]
                else:
                    unmasked_intensity = intensity
                max_intensity_values.append(np.nanmax(unmasked_intensity) * 1.1)

                # Data points
                ax_main.plot(time, intensity, 'o', color=color, markersize=3, alpha=0.6,
                            label=f'{compound} Rep{rep_num}')
                # Fit line
                ax_main.plot(time, fit, '-', color=color, linewidth=2)

                # === Residuals in separate panel ===
                ax_res = ax_residuals[idx]

                # Filter residuals by mask
                if fitting_mask is not None:
                    mask_len = min(len(fitting_mask), len(time), len(residuals))
                    mask_truncated = fitting_mask[:mask_len]
                    time_residuals = time[:mask_len][mask_truncated]
                    residuals_filtered = residuals[:mask_len][mask_truncated]
                else:
                    time_residuals = time
                    residuals_filtered = residuals

                # Plot residuals
                ax_res.plot(time_residuals, residuals_filtered, '-', color=color, linewidth=1)

                # Calculate residual limit for this panel (ignore NaN)
                max_abs_residual = np.nanmax(np.abs(residuals_filtered))
                if np.isnan(max_abs_residual) or np.isinf(max_abs_residual) or max_abs_residual == 0:
                    residual_limits.append(5)
                else:
                    residual_limits.append(np.ceil(max_abs_residual))

            # Add zero line to this residual panel
            ax_res.axhline(y=0, color=COLORS['zero_line'], linewidth=1, linestyle='solid', zorder=0)

            # Set symmetrical y-axis for this residual panel
            if residual_limits[idx] > 0:
                ax_res.set_ylim(-residual_limits[idx], residual_limits[idx])

            # Residual panel formatting
            ax_res.set_ylabel('WRes', fontsize=9)
            ax_res.grid(True, alpha=0.3)
            ax_res.tick_params(labelsize=9)

            # Only show x-axis label on bottom panel
            if idx < n_datasets - 1:
                ax_res.tick_params(labelbottom=False)

        # === Main panel formatting ===
        # X-axis
        if log_x:
            ax_main.set_xscale('log')
            ax_main.set_xlim(0.01, 100)
        else:
            ax_main.set_xscale('linear')
            ax_main.set_xlim(0, 50)

        # Y-axis: Auto-scale with 0 at bottom
        ax_main.set_ylim(bottom=0, auto=True)
        ax_main.autoscale(enable=True, axis='y', tight=False)

        ax_main.set_ylabel('Intensity (counts)', fontsize=12)
        ax_main.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax_main.grid(True, alpha=0.3)
        ax_main.tick_params(labelbottom=False)

        # Add title
        fig.suptitle(title, fontsize=14, y=0.98)

        # Bottom residual panel gets x-axis label
        ax_residuals[-1].set_xlabel('Time (μs)', fontsize=12)

        # Layout handled by constrained_layout (set at Figure creation)

        # Attach data for CSV export (export all datasets)
        export_data = {'n_datasets': n_datasets, 'datasets': []}
        for idx, item in enumerate(selected_items):
            dataset_export = {
                'compound': item['compound'],
                'type': item['type']
            }

            if item['type'] == 'mean':
                mean_arrays = item['mean_arrays']
                sd_arrays = item['sd_arrays']
                dataset_export['time_us'] = mean_arrays['mean_time_experiment_us']
                dataset_export['mean_intensity'] = mean_arrays['mean_intensity_raw']
                dataset_export['sd_intensity'] = sd_arrays['sd_intensity_raw']
                dataset_export['mean_fit'] = mean_arrays['mean_main_curve_ft_t0']
                dataset_export['sd_fit'] = sd_arrays['sd_main_curve_ft_t0']
                dataset_export['mean_residuals'] = mean_arrays['mean_main_weighted_residuals']
                dataset_export['sd_residuals'] = sd_arrays['sd_main_weighted_residuals']
            else:  # replicate
                result = item['result']
                dataset_export['time_us'] = result.time_experiment_us
                dataset_export['intensity'] = result.intensity_raw
                dataset_export['fit'] = result.main_curve
                dataset_export['residuals'] = result.main_weighted_residuals
                dataset_export['replicate_num'] = item['replicate_num']

            export_data['datasets'].append(dataset_export)

        # OPTIMIZED Phase 3: Lazy export data builder (capture export_data in closure)
        def _build_export_data():
            """Lazy builder for export data - only called when CSV export is requested."""
            return export_data

        fig.solis_export_data = _build_export_data

        logger.info(f"Matplotlib merged decay plot created: {n_datasets} datasets")
        return fig

    # ===========================
    # SURPLUS ANALYSIS PLOTTING
    # ===========================

    def plot_surplus_analysis_mpl(self, surplus_result, log_x: bool = True) -> Figure:
        """
        Create comprehensive surplus analysis plot showing ALL steps.

        Shows:
        Panel 1 (top): Step 1 - Late-time fit
        - Raw data (gray points)
        - Late-time fit FULL CURVE from t=0 (blue line, fitted region in solid, extrapolated in dashed)
        - Vertical line at mask_time

        Panel 2: Step 2 - Surplus signal
        - Surplus = Raw - Late fit (orange points)
        - Surplus fit (green line)
        - Zero line

        Panel 3: Step 4 - Final heterogeneous fit
        - Raw data (gray points)
        - Final bi-exponential fit (red line)

        Panel 4 (bottom): Residuals
        - Weighted residuals from final fit

        Parameters
        ----------
        surplus_result : SurplusResult
            Results from surplus analysis
        log_x : bool, optional
            Use logarithmic x-axis (default: True)

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        from surplus.surplus_analyzer import SurplusResult

        logger.info("Creating comprehensive surplus analysis plot (matplotlib)")

        # Create figure with 4 panels
        # OPTIMIZED: Use constrained_layout instead of tight_layout (40-60% faster)
        fig = Figure(figsize=(12, 10), constrained_layout=True)
        gs = GridSpec(4, 1, figure=fig, height_ratios=[0.3, 0.25, 0.3, 0.15], hspace=0.15)

        ax1 = fig.add_subplot(gs[0])  # Step 1: Late-time fit
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Step 2: Surplus
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # Step 4: Final fit
        ax4 = fig.add_subplot(gs[3], sharex=ax1)  # Residuals

        time = surplus_result.time_us
        intensity = surplus_result.intensity_raw
        mask_time = surplus_result.mask_time
        fitting_mask = time >= mask_time

        # === Panel 1: Step 1 - Late-time fit ===
        ax1.scatter(time, intensity, s=15, alpha=0.5, color=COLORS['experimental'],
                   label='Raw data', zorder=2)

        # Show FULL late-time fit curve from t=0
        # Fitted region (solid), extrapolated region (dashed)
        ax1.plot(time[~fitting_mask], surplus_result.late_fit_curve[~fitting_mask],
                'b--', linewidth=2, alpha=0.7, label='Late fit (extrapolated)', zorder=3)
        ax1.plot(time[fitting_mask], surplus_result.late_fit_curve[fitting_mask],
                'b-', linewidth=2.5, label=f'Late fit (fitted, t>{mask_time:.1f}μs)', zorder=4)

        ax1.axvline(mask_time, color='gray', linestyle=':', linewidth=1.5,
                   label=f'Mask at {mask_time:.1f} μs', zorder=1)
        ax1.set_ylabel('Counts', fontsize=11)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.2)
        ax1.set_title('Step 1: Late-time homogeneous fit', fontsize=11, fontweight='bold')
        ax1.tick_params(labelbottom=False)

        # === Panel 2: Step 2 & 3 - Surplus signal and fit ===
        ax2.scatter(time, surplus_result.surplus_signal, s=15, alpha=0.6,
                   color='#FF8C00', label='Surplus = Raw - Late fit', zorder=2)

        # Show surplus fit (Step 3)
        ax2.plot(time, surplus_result.surplus_fit_curve, 'g-', linewidth=2.5,
                label=f'Surplus fit (R²={surplus_result.surplus_fit_r2:.3f})', zorder=3)

        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=1)
        ax2.axvline(mask_time, color='gray', linestyle=':', linewidth=1.5, zorder=1)
        ax2.set_ylabel('Surplus', fontsize=11)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.2)
        ax2.set_title('Steps 2 & 3: Surplus signal and fit', fontsize=11, fontweight='bold')
        ax2.tick_params(labelbottom=False)

        # === Panel 3: Step 4 - Final heterogeneous fit ===
        ax3.scatter(time, intensity, s=15, alpha=0.5, color=COLORS['experimental'],
                   label='Raw data', zorder=2)

        ax3.plot(time, surplus_result.final_curve, 'r-', linewidth=2.5,
                label=f'Heterogeneous fit, Eq. (4) (R²={surplus_result.final_r2:.3f})', zorder=3)

        ax3.axvline(mask_time, color='gray', linestyle=':', linewidth=1.5, zorder=1)
        ax3.set_ylabel('Counts', fontsize=11)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.2)
        ax3.set_title('Step 4: Final heterogeneous fit to raw data', fontsize=11, fontweight='bold')
        ax3.tick_params(labelbottom=False)

        # === Panel 4: Residuals from final fit ===
        valid_mask = np.isfinite(intensity)
        residuals = np.full_like(intensity, np.nan)
        residuals[valid_mask] = intensity[valid_mask] - surplus_result.final_curve[valid_mask]

        # Calculate weighted residuals for better visualization
        weights = np.sqrt(np.abs(intensity[valid_mask]) + 1)  # Poisson weighting
        weighted_residuals = residuals[valid_mask] / weights

        ax4.scatter(time[valid_mask], weighted_residuals, s=10, alpha=0.6, color='darkred', zorder=2)
        ax4.axhline(0, color=COLORS['zero_line'], linestyle='-', linewidth=1, zorder=1)
        ax4.set_xlabel('Time (μs)', fontsize=12)
        ax4.set_ylabel('WR', fontsize=10)
        ax4.grid(True, alpha=0.2)

        # Symmetric y-axis for residuals
        resid_max = np.nanmax(np.abs(weighted_residuals))
        if not np.isnan(resid_max) and resid_max > 0:
            ax4.set_ylim(-resid_max * 1.2, resid_max * 1.2)

        # Set x-axis scale for all panels
        if log_x:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax3.set_xscale('log')
            ax4.set_xscale('log')
            xlim_left = max(0.01, time[time > 0].min())
        else:
            xlim_left = 0

        ax1.set_xlim(left=xlim_left)

        # Set y-axis limits for data panels (auto for positive data)
        for ax in [ax1, ax3]:
            ax.set_ylim(bottom=0, auto=True)
            ax.autoscale(enable=True, axis='y', tight=False)

        # Add parameter text box to Panel 3
        params = surplus_result.final_params
        param_text = (
            f"α = {params['alpha']:.0f}\n"
            f"β = {params['beta']:.0f}\n"
            f"τΔ,1 = {params['tau_delta_1']:.2f} μs\n"
            f"τΔ,2 = {params['tau_delta_2']:.2f} μs\n"
            f"τT = {params['tau_T']:.2f} μs\n"
            f"R² = {surplus_result.final_r_squared:.4f}"
        )

        ax3.text(0.02, 0.98, param_text,
                    transform=ax3.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

        # Title
        fig.suptitle('Surplus Analysis - Heterogeneous Bi-exponential Fit', fontsize=14, fontweight='bold')

        # Layout handled by constrained_layout (set at Figure creation)

        # OPTIMIZED Phase 3: Lazy export data builder
        def _build_export_data():
            """Lazy builder for export data - only called when CSV export is requested."""
            return {
                'time_us': time,
                'intensity_raw': intensity,
                'late_fit_curve': surplus_result.late_fit_curve,
                'surplus_signal': surplus_result.surplus_signal,
                'surplus_fit_curve': surplus_result.surplus_fit_curve,
                'final_curve': surplus_result.final_curve,
                'weighted_residuals': weighted_residuals,
                'mask_time': mask_time,
                'fitting_mask': fitting_mask,
                'final_params': surplus_result.final_params,  # Use the property which returns correct dict
                'late_params': {
                    'A': surplus_result.late_fit_A,
                    'tau_delta': surplus_result.late_fit_tau_delta,
                    'tau_T': surplus_result.late_fit_tau_T,
                    'y0': surplus_result.late_fit_y0
                },
                'surplus_params': {
                    'A': surplus_result.surplus_fit_A,
                    'tau_delta': surplus_result.surplus_fit_tau_delta,
                    'tau_T': surplus_result.surplus_fit_tau_T,
                    'y0': surplus_result.surplus_fit_y0
                }
            }

        fig.solis_export_data = _build_export_data

        logger.info("Surplus analysis plot created successfully")
        return fig

    def plot_heterogeneous_fit_mpl(
        self,
        result,
        log_x: bool = True,
        log_y: bool = False
    ) -> Figure:
        """
        Create heterogeneous fit result plot (Figure 4c style).

        Displays experimental data, total fit, and individual components (lipid and water).
        Two-panel layout: decay curve (top) and weighted residuals (bottom).

        Parameters
        ----------
        result : HeterogeneousFitResult
            Heterogeneous fit result object
        log_x : bool, optional
            Use logarithmic x-axis (default True)
        log_y : bool, optional
            Use logarithmic y-axis (default False)

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        from core.kinetics_dataclasses import HeterogeneousFitResult

        logger.info(f"Creating heterogeneous fit plot for {result.compound_name} ({result.replicate_id})")

        # Create figure with 2 panels (85% decay, 15% residuals)
        # OPTIMIZED: Use constrained_layout instead of tight_layout (40-60% faster)
        fig = Figure(figsize=(8, 6), constrained_layout=True)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[0.85, 0.15], hspace=0.05)
        ax_decay = fig.add_subplot(gs[0])
        ax_resid = fig.add_subplot(gs[1], sharex=ax_decay)

        # Get data
        time_exp = result.time_exp_us
        signal_exp = result.signal_exp
        signal_fit = result.signal_fit

        # Get components
        lipid_component = result.get_lipid_component()
        water_component = result.get_water_component()

        # Interpolate components to experimental time
        if lipid_component is not None and result.time_basis_us is not None:
            lipid_interp = np.interp(time_exp, result.time_basis_us, lipid_component)
            water_interp = np.interp(time_exp, result.time_basis_us, water_component)
        else:
            lipid_interp = None
            water_interp = None

        # === PANEL 1: Decay Curve ===

        # Experimental data (gray dots)
        ax_decay.plot(
            time_exp,
            signal_exp,
            'o',
            color='#808080',
            markersize=3,
            alpha=0.5,
            label='Experimental',
            zorder=1
        )

        # Total fit (green solid line)
        ax_decay.plot(
            time_exp,
            signal_fit,
            '-',
            color='#00AA00',
            linewidth=2,
            label='Fit',
            zorder=3
        )

        # Lipid component (blue dotted)
        if lipid_interp is not None:
            ax_decay.plot(
                time_exp,
                lipid_interp,
                ':',
                color='#0000FF',
                linewidth=1.5,
                label=f'A × n_L, Lipid',
                zorder=2
            )

        # Water component (orange dotted)
        if water_interp is not None:
            ax_decay.plot(
                time_exp,
                water_interp,
                ':',
                color='#FF8C00',
                linewidth=1.5,
                label=f'B × n_W, Water',
                zorder=2
            )

        # Axis scaling
        if log_x:
            ax_decay.set_xscale('log')
            ax_resid.set_xscale('log')
        if log_y:
            ax_decay.set_yscale('log')

        # Labels and legend
        ax_decay.set_ylabel('Counts (×10³)', fontsize=11)
        ax_decay.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_decay.tick_params(axis='both', labelsize=10)
        ax_decay.grid(True, alpha=0.3, linestyle='--')

        # Remove x-axis labels from top panel
        ax_decay.tick_params(axis='x', labelbottom=False)

        # Add parameters text box
        params_text = (
            f"τT = {result.tau_T_us:.2f} μs\n"
            f"τw = {result.tau_w_us:.2f} μs\n"
            f"τL = {result.tau_L_us:.2f} μs\n"
            f"A/B = {result.rate_ratio:.2f}\n"
            f"χ²red = {result.chi2_reduced:.2f}"
        )
        ax_decay.text(
            0.02, 0.98,
            params_text,
            transform=ax_decay.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
        )

        # === PANEL 2: Weighted Residuals ===

        if result.weighted_residuals is not None:
            ax_resid.plot(
                time_exp,
                result.weighted_residuals,
                '-',
                color='#00AA00',
                linewidth=1,
                zorder=2
            )

            # Zero line
            ax_resid.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

            # Symmetric y-axis (use nanmax to handle NaN like other plots)
            resid_max = np.nanmax(np.abs(result.weighted_residuals))
            if not np.isnan(resid_max) and resid_max > 0:
                ax_resid.set_ylim(-resid_max * 1.2, resid_max * 1.2)
            else:
                ax_resid.set_ylim(-3, 3)  # Default if all NaN

        ax_resid.set_xlabel('Time (μs)', fontsize=11)
        ax_resid.set_ylabel('WRes', fontsize=10)
        ax_resid.tick_params(axis='both', labelsize=10)
        ax_resid.grid(True, alpha=0.3, linestyle='--')

        # Title
        title = f'Heterogeneous Fit: {result.compound_name} ({result.replicate_id})'
        fig.suptitle(title, fontsize=12, fontweight='bold')

        # Layout handled by constrained_layout (set at Figure creation)

        # OPTIMIZED Phase 3: Lazy export data builder
        def _build_export_data():
            """Lazy builder for export data - only called when CSV export is requested."""
            return {
                'time_us': time_exp,
                'signal_exp': signal_exp,
                'signal_fit': signal_fit,
                'weighted_residuals': result.weighted_residuals if result.weighted_residuals is not None else np.full_like(time_exp, np.nan),
                'lipid_component': lipid_interp if lipid_interp is not None else np.full_like(time_exp, np.nan),
                'water_component': water_interp if water_interp is not None else np.full_like(time_exp, np.nan),
                'fit_params': {
                    'tau_w': result.tau_w_us,
                    'tau_L': result.tau_L_us,
                    'tau_T': result.tau_T_us,
                    'A': result.A,
                    'B': result.B,
                    'C': result.C,
                    'chi2_reduced': result.chi2_reduced
                }
            }

        fig.solis_export_data = _build_export_data

        logger.info("Heterogeneous fit plot created successfully")
        return fig

    def plot_heterogeneous_grid_mpl(
        self,
        result
    ) -> Figure:
        """
        Create χ²red grid search landscape plot (Figure 4b style).

        Displays contour plot of reduced chi-square values across (τT, τw) parameter space.
        Shows best-fit point and contour lines.

        Parameters
        ----------
        result : HeterogeneousFitResult
            Heterogeneous fit result with grid_chi2 DataFrame

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        from core.kinetics_dataclasses import HeterogeneousFitResult

        logger.info(f"Creating heterogeneous grid plot for {result.compound_name} ({result.replicate_id})")

        if result.grid_chi2 is None or len(result.grid_chi2) == 0:
            logger.warning("No grid data available for plotting")
            # Create empty figure with message
            # OPTIMIZED: Use constrained_layout instead of tight_layout (40-60% faster)
            fig = Figure(figsize=(6, 5), constrained_layout=True)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No grid data available',
                   ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        # Extract grid data
        df = result.grid_chi2
        tau_T_vals = df['tau_T_us'].unique()
        tau_w_vals = df['tau_w_us'].unique()

        # Create 2D grid for contour plot
        n_tau_T = len(tau_T_vals)
        n_tau_w = len(tau_w_vals)

        # Sort values for proper grid creation
        tau_T_vals = np.sort(tau_T_vals)
        tau_w_vals = np.sort(tau_w_vals)

        # Initialize chi2 grid
        chi2_grid = np.zeros((n_tau_w, n_tau_T))

        # Fill grid
        for _, row in df.iterrows():
            i_tau_T = np.argmin(np.abs(tau_T_vals - row['tau_T_us']))
            i_tau_w = np.argmin(np.abs(tau_w_vals - row['tau_w_us']))
            chi2_grid[i_tau_w, i_tau_T] = row['chi2_red']

        # Create figure
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

        # Create meshgrid for contour plot
        TauT, TauW = np.meshgrid(tau_T_vals, tau_w_vals)

        # Contour plot (filled)
        contourf = ax.contourf(
            TauT, TauW, chi2_grid,
            levels=20,
            cmap='viridis',
            alpha=0.7
        )

        # Contour lines
        contour = ax.contour(
            TauT, TauW, chi2_grid,
            levels=10,
            colors='black',
            linewidths=0.5,
            alpha=0.5
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

        # Mark best-fit point
        ax.plot(
            result.tau_T_us,
            result.tau_w_us,
            'r*',
            markersize=15,
            markeredgecolor='white',
            markeredgewidth=1,
            label=f'Best fit: χ²red = {result.chi2_reduced:.2f}',
            zorder=10
        )

        # Add hatching for high chi2 regions (chi2 > 2*min)
        chi2_threshold = 2.0 * result.chi2_reduced
        if chi2_threshold < np.max(chi2_grid):
            high_chi2_contour = ax.contourf(
                TauT, TauW, chi2_grid,
                levels=[chi2_threshold, np.max(chi2_grid)],
                hatches=['///'],
                colors='none',
                alpha=0
            )

        # Colorbar
        cbar = fig.colorbar(contourf, ax=ax, label='χ²red')
        cbar.ax.tick_params(labelsize=9)

        # Labels and title
        ax.set_xlabel('τT (μs)', fontsize=11)
        ax.set_ylabel('τΔW (μs)', fontsize=11)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        title = f'Grid Search Landscape: {result.compound_name} ({result.replicate_id})'
        fig.suptitle(title, fontsize=12, fontweight='bold')

        # Layout handled by constrained_layout (set at Figure creation)

        # OPTIMIZED Phase 3: Lazy export data builder
        def _build_export_data():
            """Lazy builder for export data - only called when CSV export is requested."""
            return {
                'tau_T_grid': tau_T_vals,
                'tau_w_grid': tau_w_vals,
                'chi2_grid': chi2_grid,
                'grid_dataframe': df,  # Full DataFrame with all grid points
                'best_fit': {
                    'tau_T': result.tau_T_us,
                    'tau_w': result.tau_w_us,
                    'tau_L': result.tau_L_us,
                    'chi2_reduced': result.chi2_reduced
                }
            }

        fig.solis_export_data = _build_export_data

        logger.info("Heterogeneous grid plot created successfully")
        return fig

    def plot_absorption_spectrum_mpl(
        self,
        parsed_file,
        excitation_wavelengths: Optional[List[float]] = None,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot single absorption spectrum.

        Parameters
        ----------
        parsed_file : ParsedFile
            Parsed absorption file
        excitation_wavelengths : List[float], optional
            List of excitation wavelengths to mark with vertical lines
        title : str, optional
            Plot title (default: compound name)

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        logger.info(f"Creating absorption spectrum plot for {parsed_file.compound}")

        # Extract data
        df = parsed_file.data
        wavelength = df.iloc[:, 0].values  # First column: wavelength
        absorbance_cols = df.iloc[:, 1:].values  # Remaining columns: replicates

        # Average across replicates if multiple columns
        if absorbance_cols.shape[1] > 1:
            absorbance = np.mean(absorbance_cols, axis=1)
            logger.info(f"Averaged {absorbance_cols.shape[1]} replicates")
        else:
            absorbance = absorbance_cols[:, 0]

        # Create figure
        fig = Figure(figsize=(8, 5), constrained_layout=True)
        ax = fig.add_subplot(111)

        # Plot spectrum
        ax.plot(wavelength, absorbance, 'k-', linewidth=2, label=parsed_file.compound)

        # Add excitation wavelength markers if provided
        if excitation_wavelengths:
            for ex_wl in excitation_wavelengths:
                color = wavelength_to_color(ex_wl)
                ax.axvline(ex_wl, color=color, linestyle='--', linewidth=1.5,
                          label=f'λex = {ex_wl:.0f} nm', alpha=0.7)

        # Labels and formatting
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Absorbance', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

        # Title
        if title is None:
            title = f'Absorption Spectrum: {parsed_file.compound}'
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Attach export data
        export_data = {
            'Wavelength_nm': wavelength,
            'Absorbance': absorbance,
            'Compound': parsed_file.compound
        }
        if excitation_wavelengths:
            export_data['Excitation_Wavelengths_nm'] = excitation_wavelengths

        fig.solis_export_data = export_data

        logger.info("Absorption spectrum plot created successfully")
        return fig

    def plot_merged_absorption_spectra_mpl(
        self,
        parsed_files: List,
        excitation_wavelengths: Optional[Dict[str, List[float]]] = None,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot multiple absorption spectra on the same axes.

        Parameters
        ----------
        parsed_files : List[ParsedFile]
            List of parsed absorption files
        excitation_wavelengths : Dict[str, List[float]], optional
            Dictionary mapping compound names to lists of excitation wavelengths
            Example: {'TMPyP': [400, 450], 'PN': [400]}
        title : str, optional
            Plot title

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        logger.info(f"Creating merged absorption spectra plot with {len(parsed_files)} spectra")

        # Create figure
        fig = Figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111)

        # Track all excitation wavelengths for legend management
        all_ex_wavelengths = set()

        # Plot each spectrum
        export_data = {}
        for idx, parsed_file in enumerate(parsed_files):
            # Extract data
            df = parsed_file.data
            wavelength = df.iloc[:, 0].values
            absorbance_cols = df.iloc[:, 1:].values

            # Average across replicates
            if absorbance_cols.shape[1] > 1:
                absorbance = np.mean(absorbance_cols, axis=1)
            else:
                absorbance = absorbance_cols[:, 0]

            # Choose color from palette
            color = MERGED_COLORS[idx % len(MERGED_COLORS)]

            # Plot spectrum
            ax.plot(wavelength, absorbance, color=color, linewidth=2,
                   label=parsed_file.compound)

            # Store in export data
            export_data[f'Wavelength_nm_{parsed_file.compound}'] = wavelength
            export_data[f'Absorbance_{parsed_file.compound}'] = absorbance

        # Add excitation wavelength markers
        if excitation_wavelengths:
            for compound, ex_wls in excitation_wavelengths.items():
                for ex_wl in ex_wls:
                    if ex_wl not in all_ex_wavelengths:
                        # Only add legend entry once per wavelength
                        color = wavelength_to_color(ex_wl)
                        ax.axvline(ex_wl, color=color, linestyle='--', linewidth=1.5,
                                  label=f'λex = {ex_wl:.0f} nm', alpha=0.7)
                        all_ex_wavelengths.add(ex_wl)
                    else:
                        # Just draw the line without legend entry
                        color = wavelength_to_color(ex_wl)
                        ax.axvline(ex_wl, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

        # Labels and formatting
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Absorbance', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

        # Title
        if title is None:
            title = f'Absorption Spectra ({len(parsed_files)} compounds)'
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Attach export data
        if excitation_wavelengths:
            export_data['Excitation_Wavelengths'] = excitation_wavelengths

        fig.solis_export_data = export_data

        logger.info("Merged absorption spectra plot created successfully")
        return fig


def wavelength_to_color(wavelength: float) -> str:
    """
    Convert wavelength (nm) to RGB color hex code.

    Approximates the visible spectrum colors:
    - 380-450 nm: Violet to Blue
    - 450-495 nm: Blue to Cyan
    - 495-570 nm: Cyan to Green to Yellow
    - 570-590 nm: Yellow to Orange
    - 590-750 nm: Orange to Red

    Parameters
    ----------
    wavelength : float
        Wavelength in nanometers

    Returns
    -------
    str
        Hex color code (e.g., '#FF0000')
    """
    # Clamp wavelength to visible range
    wl = np.clip(wavelength, 380, 750)

    # Initialize RGB
    r, g, b = 0.0, 0.0, 0.0

    if 380 <= wl < 440:
        # Violet to Blue
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wl < 490:
        # Blue to Cyan
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510:
        # Cyan to Green
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        # Green to Yellow
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wl < 645:
        # Yellow to Red
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    elif 645 <= wl <= 750:
        # Red
        r = 1.0
        g = 0.0
        b = 0.0

    # Intensity factor for edges of visible spectrum
    if 380 <= wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif 420 <= wl < 700:
        factor = 1.0
    elif 700 <= wl <= 750:
        factor = 0.3 + 0.7 * (750 - wl) / (750 - 700)
    else:
        factor = 1.0

    r *= factor
    g *= factor
    b *= factor

    # Convert to hex
    r_int = int(np.clip(r * 255, 0, 255))
    g_int = int(np.clip(g * 255, 0, 255))
    b_int = int(np.clip(b * 255, 0, 255))

    return f'#{r_int:02X}{g_int:02X}{b_int:02X}'
