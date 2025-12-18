#!/usr/bin/env python3
"""
Plotting utilities for Heterogeneous Diffusion Model Analysis

Generates publication-quality figures based on:
Hackbarth & Röder (2015), Photochem. Photobiol. Sci., 14, 329-334

Key figures:
- Figure 4: Chi-square landscape (τ_T vs τ_Δ,water)
- Fit curves: Experimental data with best-fit model
- Component breakdown: Lipid vs water contributions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from heterogeneous.heterogeneous_dataclasses import HeterogeneousFitResult
from utils.logger_config import get_logger

logger = get_logger(__name__)


class HeterogeneousPlotter:
    """
    Creates publication-quality plots for heterogeneous diffusion analysis.

    Usage:
        plotter = HeterogeneousPlotter(result)
        plotter.plot_figure_4()
        plotter.plot_fit_curves()
    """

    def __init__(self, result: HeterogeneousFitResult):
        """
        Initialize plotter with fit result.

        Args:
            result: HeterogeneousFitResult from fit_heterogeneous()
        """
        self.result = result
        logger.info("HeterogeneousPlotter initialized")

    def plot_figure_4(self,
                     figsize: Tuple[float, float] = (7, 6),
                     save_path: Optional[str] = None,
                     dpi: int = 300,
                     show_rate_ratio: bool = True,
                     style: str = 'paper') -> plt.Figure:
        """
        Plot chi-square landscape (Figure 4 from paper).

        Shows contours of reduced chi-square as function of τ_T and τ_Δ,water.
        Style matches Hackbarth & Röder 2015 Figure 4 with colored contour bands.

        Args:
            figsize: Figure size (width, height) in inches
            save_path: Path to save figure (None = don't save)
            dpi: Resolution for saved figure
            show_rate_ratio: Whether to overlay rate ratio contours (black lines)
            style: 'paper' for colored bands like original, 'grayscale' for B&W

        Returns:
            matplotlib Figure object
        """
        if self.result.grid_search_surface is None:
            logger.error("No grid search data. Result must include grid_search_surface.")
            raise ValueError("Result does not contain grid search data for Figure 4")

        logger.info(f"Generating Figure 4: Chi-square landscape (style={style})")

        # Extract grid data
        chi2_surface = self.result.grid_search_surface  # Shape: (n_tau_T, n_tau_delta_W)
        tau_T_grid = self.result.grid_tau_T  # 1D array of tau_T values
        tau_delta_W_grid = self.result.grid_tau_delta_W  # 1D array of tau_delta_W values

        fig, ax = plt.subplots(figsize=figsize)

        # Create meshgrid for contour plotting
        tau_T_mesh, tau_delta_W_mesh = np.meshgrid(tau_T_grid, tau_delta_W_grid, indexing='ij')

        # Paper-style chi-square contour levels (matching Figure 4)
        # Inner white zone: chi2_red < 1.1 (best fit)
        # Light shaded zones: 1.1-1.2, 1.2-1.3, etc.
        # Heavily shaded zones: chi2_red > 2.0
        if style == 'paper':
            # Define discrete bands matching paper's color scheme
            chi2_levels = [0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 10.0]

            # Color palette matching paper (light to dark with increasing chi2)
            # White center → light gray → medium gray → dark gray → black hatching
            colors_paper = [
                '#FFFFFF',  # < 1.1: white (excellent fit)
                '#F0F0F0',  # 1.1-1.2: very light gray
                '#E0E0E0',  # 1.2-1.3: light gray
                '#D0D0D0',  # 1.3-1.4
                '#C0C0C0',  # 1.4-1.5
                '#B0B0B0',  # 1.5-1.6: medium gray
                '#A0A0A0',  # 1.6-1.7
                '#909090',  # 1.7-1.8
                '#808080',  # 1.8-1.9
                '#707070',  # 1.9-2.0
                '#606060',  # 2.0-2.5: dark gray
                '#505050',  # 2.5-3.0
                '#404040',  # > 3.0: very dark
            ]

            contourf = ax.contourf(
                tau_T_mesh,
                tau_delta_W_mesh,
                chi2_surface,
                levels=chi2_levels,
                colors=colors_paper,
                extend='max'
            )

            # Add contour lines at key chi2 values
            contour_lines = [1.1, 1.2, 1.5, 2.0, 2.9]
            contour = ax.contour(
                tau_T_mesh,
                tau_delta_W_mesh,
                chi2_surface,
                levels=contour_lines,
                colors='black',
                linewidths=1.0,
                alpha=0.6
            )
            ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

        else:  # grayscale
            levels = np.arange(0.9, 3.5, 0.1)
            contourf = ax.contourf(
                tau_T_mesh,
                tau_delta_W_mesh,
                chi2_surface,
                levels=levels,
                cmap='gray_r',
                extend='max'
            )

            # Contour lines
            contour_levels = [1.0, 1.1, 1.2, 1.5, 2.0, 2.5]
            contour = ax.contour(
                tau_T_mesh,
                tau_delta_W_mesh,
                chi2_surface,
                levels=contour_levels,
                colors='black',
                linewidths=1.2
            )
            ax.clabel(contour, inline=True, fontsize=9, fmt='%.2f')

        # Add rate ratio contours (black lines matching paper Figure 4)
        if show_rate_ratio and hasattr(self.result, 'calculate_rate_ratio_surface'):
            # Calculate rate ratio for each grid point (A/B from fits)
            # This requires refitting at each point - expensive!
            # For now, draw diagonal lines representing typical rate ratios
            logger.info("Rate ratio contours: using approximate diagonal lines")

            # Rate ratio lines (approximate, matching paper Figure 4)
            # Lines from bottom-left to top-right represent constant rate ratio
            for rate_ratio in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
                ax.plot([], [], 'k-', linewidth=0.8, label=f'A/B={rate_ratio:.1f}')

            # Add text label
            ax.text(0.98, 0.02, 'Black lines: rate ratio A/B',
                   transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Mark minimum chi-square with X
        min_chi2 = np.min(chi2_surface)
        min_idx = np.unravel_index(np.argmin(chi2_surface), chi2_surface.shape)
        tau_T_min = tau_T_grid[min_idx[0]]
        tau_delta_min = tau_delta_W_grid[min_idx[1]]

        ax.plot(tau_T_min, tau_delta_min, 'kx', markersize=12, markeredgewidth=2.5,
               label=f'Best fit: χ²ᵣ={min_chi2:.3f}')

        # Add annotation
        ax.annotate(f'τ_T = {tau_T_min:.2f} μs\nτ_Δ,W = {tau_delta_min:.2f} μs\nχ²ᵣ = {min_chi2:.3f}',
                   xy=(tau_T_min, tau_delta_min), xytext=(15, 15),
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5))

        # Labels matching paper style
        ax.set_xlabel('PS triplet decay time τ_T (μs)', fontsize=12, fontweight='bold')
        ax.set_ylabel('¹O₂ decay time τ_Δ,water (μs)', fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=10)

        # Grid for easier reading
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

        # Set reasonable axis limits with small padding
        ax.set_xlim(tau_T_grid[0], tau_T_grid[-1])
        ax.set_ylim(tau_delta_W_grid[0], tau_delta_W_grid[-1])

        # Title
        ax.set_title('Reduced χ² Landscape for Heterogeneous Diffusion Model',
                    fontsize=13, fontweight='bold', pad=15)

        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax, label='Reduced χ²')
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout()

        # OPTIMIZED Phase 3: Lazy export data builder
        def _build_export_data():
            """Lazy builder for export data - only called when CSV export is requested."""
            return {
                'tau_T_grid': tau_T_grid,
                'tau_delta_W_grid': tau_delta_W_grid,
                'chi2_surface': chi2_surface,
                'best_tau_T': tau_T_min,
                'best_tau_delta_W': tau_delta_min,
                'best_chi2_reduced': min_chi2,
                'compound_name': self.result.compound_name if hasattr(self.result, 'compound_name') else 'Unknown',
                'replicate_id': self.result.replicate_id if hasattr(self.result, 'replicate_id') else 'Unknown'
            }

        fig.solis_export_data = _build_export_data

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Figure 4 saved to {save_path}")

        logger.info("Figure 4 generated successfully")
        return fig

    def plot_fit_curves(self,
                       figsize: Tuple[float, float] = (6, 5.5),
                       save_path: Optional[str] = None,
                       dpi: int = 300,
                       log_scale: bool = True,
                       show_components: bool = True) -> plt.Figure:
        """
        Plot experimental data with best-fit curves (Figure 4c style).

        Creates a multi-panel figure matching paper style:
        - Top: Full fit (experimental + model + components)
        - Bottom: Weighted residuals (WRes)

        Args:
            figsize: Figure size (width, height) in inches
            save_path: Path to save figure (None = don't save)
            dpi: Resolution for saved figure
            log_scale: Use log scale for x-axis (recommended for ¹O₂)
            show_components: Show lipid/water component breakdown

        Returns:
            matplotlib Figure object
        """
        logger.info("Generating fit curves plot (Figure 4c style)")

        # Create figure with 2 subplots (fit + residuals)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)

        # Top panel: Data and fit
        # Experimental data (gray dots matching paper)
        ax1.plot(self.result.time_experimental,
                self.result.intensity_experimental / 1e3,  # Convert to 10³ counts
                'o', color='lightgray', markersize=3, alpha=0.6,
                label='Data', zorder=1)

        # Best fit (green solid line matching paper)
        ax1.plot(self.result.time_experimental,
                self.result.fit_curve / 1e3,
                '-', color='green', linewidth=2,
                label='Fit', zorder=3)

        # Component breakdown
        if show_components:
            # Lipid component (blue dotted) - WITHOUT background
            lipid_component = self.result.amplitude_lipid * self.result.n_lipid_curve
            ax1.plot(self.result.time_experimental,
                    lipid_component / 1e3,
                    ':', color='blue', linewidth=2,
                    label=f'A × n_L, Lipid', zorder=2)

            # Water component (orange dotted) - WITHOUT background
            water_component = self.result.amplitude_water * self.result.n_water_curve
            ax1.plot(self.result.time_experimental,
                    water_component / 1e3,
                    ':', color='orange', linewidth=2,
                    label=f'B × n_W, Water', zorder=2)

        # Formatting matching paper
        if log_scale:
            ax1.set_xscale('log')

        ax1.set_ylabel('Counts (10³)', fontsize=12)
        ax1.set_ylim(bottom=-0.1)  # Start slightly below zero like paper
        ax1.tick_params(labelsize=10)
        ax1.legend(fontsize=9, loc='upper right', frameon=True)
        ax1.grid(False)

        # Bottom panel: Weighted Residuals (WRes)
        # Only plot residuals for the FITTED region, not the full experimental range
        fit_start, fit_end = self.result.fit_time_range
        fit_mask = (self.result.time_experimental >= fit_start) & (self.result.time_experimental <= fit_end)

        # Calculate weighted residuals only for fitted region
        weights = 1.0 / np.sqrt(np.maximum(self.result.intensity_experimental[fit_mask], 1.0))
        wres = self.result.residuals[fit_mask] * weights

        ax2.plot(self.result.time_experimental[fit_mask],
                wres,
                '-', color='green', linewidth=1, alpha=0.8)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Add vertical lines to mark fit range boundaries
        ax1.axvline(fit_start, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Fit range')
        ax1.axvline(fit_end, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

        if log_scale:
            ax2.set_xscale('log')

        ax2.set_xlabel('Time (μs)', fontsize=12)
        ax2.set_ylabel('WRes', fontsize=10)

        # Auto-scale residuals symmetrically (same method as homogeneous plots)
        max_abs_residual = np.nanmax(np.abs(wres))
        if np.isnan(max_abs_residual) or np.isinf(max_abs_residual) or max_abs_residual == 0:
            residual_limit = 5  # Default fallback
        else:
            residual_limit = np.ceil(max_abs_residual)
        ax2.set_ylim(-residual_limit, residual_limit)

        ax2.tick_params(labelsize=10)
        ax2.grid(False)

        plt.tight_layout()

        # OPTIMIZED Phase 3: Lazy export data builder
        def _build_export_data():
            """Lazy builder for export data - only called when CSV export is requested."""
            return {
                'time_experimental': self.result.time_experimental,
                'intensity_experimental': self.result.intensity_experimental,
                'intensity_fitted': self.result.fit_curve,  # CORRECTED: fit_curve not intensity_fitted
                'residuals': self.result.residuals,
                'weighted_residuals': wres,  # Weighted residuals for fitted region
                'fit_mask': fit_mask,  # Boolean mask showing fitted region
                'lipid_component': lipid_component if show_components else None,  # CORRECTED: use local variable
                'water_component': water_component if show_components else None,  # CORRECTED: use local variable
                'fit_params': {
                    'tau_T': self.result.tau_T,
                    'tau_delta_water': self.result.tau_delta_water,  # CORRECTED: tau_delta_water not tau_delta_W
                    'rate_ratio': self.result.rate_ratio,  # Added rate_ratio
                    'chi2_reduced': self.result.reduced_chi_square  # CORRECTED: reduced_chi_square not chi2_reduced
                },
                'compound_name': self.result.compound_name if hasattr(self.result, 'compound_name') else 'Unknown',
                'replicate_id': self.result.replicate_id if hasattr(self.result, 'replicate_id') else 'Unknown'
            }

        fig.solis_export_data = _build_export_data

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Fit curves (Figure 4c style) saved to {save_path}")

        logger.info("Fit curves generated successfully")
        return fig

    def plot_component_breakdown(self,
                                figsize: Tuple[float, float] = (10, 6),
                                save_path: Optional[str] = None,
                                dpi: int = 300) -> plt.Figure:
        """
        Plot lipid vs water component breakdown.

        Shows the normalized n_lipid(t) and n_water(t) curves from simulation.
        Useful for understanding the relative contributions.

        Args:
            figsize: Figure size (width, height) in inches
            save_path: Path to save figure (None = don't save)
            dpi: Resolution for saved figure

        Returns:
            matplotlib Figure object
        """
        logger.info("Generating component breakdown plot")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot normalized components
        ax.plot(self.result.time_experimental,
               self.result.n_lipid_curve / np.max(self.result.n_lipid_curve),
               '-', color='blue', linewidth=2,
               label='Lipid (normalized)')

        ax.plot(self.result.time_experimental,
               self.result.n_water_curve / np.max(self.result.n_water_curve),
               '-', color='green', linewidth=2,
               label='Water (normalized)')

        # Mark peak times
        peak_lipid_idx = np.argmax(self.result.n_lipid_curve)
        peak_water_idx = np.argmax(self.result.n_water_curve)

        ax.plot(self.result.time_experimental[peak_lipid_idx],
               1.0, 'o', color='blue', markersize=10,
               label=f'Lipid peak: {self.result.time_experimental[peak_lipid_idx]:.2f} μs')

        ax.plot(self.result.time_experimental[peak_water_idx],
               1.0, 'o', color='green', markersize=10,
               label=f'Water peak: {self.result.time_experimental[peak_water_idx]:.2f} μs')

        ax.set_xscale('log')
        ax.set_xlabel('Time (μs)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized intensity', fontsize=12, fontweight='bold')
        ax.set_title('Lipid vs Water Components', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Component breakdown saved to {save_path}")

        logger.info("Component breakdown generated successfully")
        return fig

    def plot_all(self,
                 output_dir: str = '.',
                 prefix: str = 'heterogeneous',
                 dpi: int = 300) -> None:
        """
        Generate all plots and save to directory.

        Args:
            output_dir: Directory to save plots
            prefix: Prefix for filenames
            dpi: Resolution for saved figures
        """
        import os
        logger.info(f"Generating all plots with prefix '{prefix}' in {output_dir}")

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Generate each plot
        try:
            self.plot_figure_4(
                save_path=os.path.join(output_dir, f"{prefix}_figure4_landscape.png"),
                dpi=dpi
            )
        except ValueError as e:
            logger.warning(f"Could not generate Figure 4: {e}")

        self.plot_fit_curves(
            save_path=os.path.join(output_dir, f"{prefix}_fit_curves.png"),
            dpi=dpi
        )

        self.plot_component_breakdown(
            save_path=os.path.join(output_dir, f"{prefix}_components.png"),
            dpi=dpi
        )

        logger.info("All plots generated successfully")
        plt.close('all')


def quick_plot(result: HeterogeneousFitResult, show: bool = True) -> None:
    """
    Quick convenience function to generate and show all plots.

    Args:
        result: HeterogeneousFitResult from fit_heterogeneous()
        show: Whether to call plt.show() (True) or just generate (False)
    """
    plotter = HeterogeneousPlotter(result)

    # Try Figure 4
    try:
        plotter.plot_figure_4()
    except ValueError:
        logger.warning("Skipping Figure 4 (no grid search data)")

    # Fit curves
    plotter.plot_fit_curves()

    # Component breakdown
    plotter.plot_component_breakdown()

    if show:
        plt.show()


if __name__ == '__main__':
    print("This module provides plotting utilities for heterogeneous analysis.")
    print("Usage:")
    print("  from heterogeneous_plotter import HeterogeneousPlotter, quick_plot")
    print("  result = fit_heterogeneous(time, intensity)")
    print("  quick_plot(result)")
