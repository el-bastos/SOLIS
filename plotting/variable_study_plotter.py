"""
Variable Study Plotter

Creates plots for studying relationships between amplitude (α/A parameter)
and absorbance or excitation intensity.

Includes:
- Combined emission plots with multiple datasets
- α vs (1 - 10^(-A(λ))) correlation plots
- α vs Excitation Intensity correlation plots
- Linear regression with forced zero intercept
- Outlier detection and linear range identification
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from scipy import stats

# Matplotlib imports for Session 15
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from utils.logger_config import get_logger
logger = get_logger(__name__)


# === Color Scheme ===
COLORS = {
    'datasets': [
        '#CC0000',  # Red
        '#0066CC',  # Blue
        '#00AA00',  # Green
        '#FF8C00',  # Orange
        '#9900CC',  # Purple
        '#CC6600',  # Brown
        '#00CCCC',  # Cyan
        '#CC0099',  # Magenta
    ],
    'linear_points': '#00AA00',  # Green for points in linear range
    'outlier_points': '#CC0000',  # Red for outliers
    'regression_line': '#000000',  # Black for regression line
    'grid': 'rgba(200, 200, 200, 0.3)',
    'axis_line': 'rgba(0, 0, 0, 0.5)',
}


def perform_linear_regression_through_origin(
    x: np.ndarray,
    y: np.ndarray,
    confidence_level: float = 0.90,
    min_r_squared: float = 0.95,
    min_linear_points: int = 2
) -> Dict[str, Any]:
    """
    Simple linear regression through origin (0,0) and lowest data point.

    Fit a line through exactly two points:
    1. Origin (0, 0)
    2. The data point with the lowest X value

    All other points are shown but not used for fitting.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (absorption or intensity)
    y : np.ndarray
        Dependent variable (amplitude α)
    confidence_level : float
        (Unused, kept for API compatibility)
    min_r_squared : float
        (Unused, kept for API compatibility)
    min_linear_points : int
        (Unused, kept for API compatibility)

    Returns
    -------
    dict
        Contains: slope, intercept (always 0), r_squared (always 1.0 for 2-point fit),
        residuals, linear_mask, saturation_mask, success
    """
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if len(x_clean) < 1:
        logger.warning("Insufficient data points for regression")
        return {
            'slope': None,
            'intercept': 0.0,
            'r_squared': None,
            'residuals': None,
            'linear_mask': None,
            'saturation_mask': None,
            'n_total': 0,
            'n_linear': 0,
            'n_saturation': 0,
            'x_linear_max': None,
            'model_type': 'two_point',
            'warning': 'Insufficient data',
            'success': False
        }

    n_total = len(x_clean)

    # Find the point with the LOWEST x value
    min_idx = np.argmin(x_clean)
    x_min = x_clean[min_idx]
    y_min = y_clean[min_idx]

    # Calculate slope from origin (0, 0) through lowest point
    slope = y_min / x_min if x_min != 0 else 0

    logger.info(f"Simple two-point linear fit: Origin (0, 0) + Lowest point ({x_min:.3f}, {y_min:.1f})")
    logger.info(f"  Slope: {slope:.2e}")
    logger.info(f"  All {n_total} data points shown, only lowest point used for fit")

    # Calculate predictions and residuals for ALL points
    y_pred_all = slope * x_clean
    residuals_all = y_clean - y_pred_all

    # For R² calculation with 2-point fit through origin, use lowest point only
    # R² is always 1.0 for a perfect 2-point fit
    r_squared = 1.0

    # Create masks: mark ALL valid points as "linear" (no saturation distinction)
    # Since we're not testing linearity, all points are just data points
    linear_mask = valid_mask.copy()  # All valid points are shown as data
    saturation_mask = np.zeros(len(x), dtype=bool)  # No saturation marking

    # Residuals in original order
    residuals_original = np.zeros(len(x))
    residuals_original[valid_mask] = residuals_all

    logger.info(f"Two-point fit complete: slope={slope:.2e}, {n_total} data points shown")

    return {
        'slope': slope,
        'intercept': 0.0,  # Always zero (line through origin)
        'r_squared': r_squared,  # Always 1.0 for 2-point fit
        'residuals': residuals_original,
        'residual_std': 0.0,  # Not meaningful for 2-point fit
        'linear_mask': linear_mask,  # All valid points
        'saturation_mask': saturation_mask,  # Empty (no saturation marking)
        'n_total': n_total,
        'n_linear': n_total,  # All points shown as data
        'n_saturation': 0,  # No saturation distinction
        'x_linear_max': x_min,
        'se_slope': None,  # Not calculable for 2-point fit
        'prediction_interval_width': None,  # Not applicable
        'confidence_level': confidence_level,
        'model_type': 'two_point',
        'warning': None,
        'success': True
    }


# === MATPLOTLIB VERSIONS ===

def plot_alpha_vs_absorption_mpl(
    selected_items: List[Dict[str, Any]],
    ei_unit: str = "a.u."
) -> Tuple[Figure, Dict[str, Any]]:
    """
    Create α vs (1 - 10^(-A(λ))) correlation plot with linear regression using matplotlib.

    Parameters
    ----------
    selected_items : list of dict
        List of selected items from data browser
    ei_unit : str
        Unit string (not used here, kept for API consistency)

    Returns
    -------
    tuple
        (fig, regression_stats): Matplotlib figure and regression statistics dict
    """
    # Extract data (reuse same logic as Plotly version)
    abs_values = []
    alpha_values = []
    alpha_errors = []
    labels = []

    for item in selected_items:
        abs_str = item.get('abs_at_wavelength', '')
        if abs_str == '—' or not abs_str:
            continue

        try:
            abs_value = float(abs_str)
        except ValueError:
            continue

        analysis_result = item.get('analysis_result')
        if analysis_result is None:
            continue

        try:
            if isinstance(analysis_result, list):
                A_vals = [r.parameters.A for r in analysis_result if hasattr(r, 'parameters')]
                if A_vals:
                    alpha_mean = np.mean(A_vals)
                    alpha_sd = np.std(A_vals, ddof=1) if len(A_vals) > 1 else 0
                else:
                    continue
            elif isinstance(analysis_result, dict):
                if 'statistics' in analysis_result:
                    stats = analysis_result['statistics']
                    alpha_mean = stats.get('A_mean')
                    alpha_sd = stats.get('A_sd', 0)
                elif 'results' in analysis_result:
                    results_list = analysis_result['results']
                    A_vals = [r.parameters.A for r in results_list if hasattr(r, 'parameters')]
                    if A_vals:
                        alpha_mean = np.mean(A_vals)
                        alpha_sd = np.std(A_vals, ddof=1) if len(A_vals) > 1 else 0
                    else:
                        continue
                else:
                    continue

                if alpha_mean is None or alpha_mean == 'ND':
                    continue
            else:
                if hasattr(analysis_result, 'parameters'):
                    alpha_mean = analysis_result.parameters.A
                    alpha_sd = 0
                else:
                    continue

            abs_values.append(abs_value)
            alpha_values.append(alpha_mean)
            alpha_errors.append(alpha_sd)
            compound = item.get('compound', 'Unknown')
            labels.append(compound)

        except Exception as e:
            logger.warning(f"Error extracting data: {e}")
            continue

    if len(abs_values) < 2:
        raise ValueError("Need at least 2 data points with A(λ) and α values")

    abs_array = np.array(abs_values)
    alpha_array = np.array(alpha_values)
    alpha_err_array = np.array(alpha_errors)

    # Calculate x = 1 - 10^(-A(λ))
    x_absorbed = 1 - 10**(-abs_array)

    # Perform linear regression
    regression_stats = perform_linear_regression_through_origin(x_absorbed, alpha_array)

    if not regression_stats['success']:
        raise ValueError("Linear regression failed")

    # Create matplotlib figure
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)

    # Plot all data points (no linear/saturation distinction)
    linear_mask = regression_stats['linear_mask']

    # Plot all points with same style
    if np.any(linear_mask):
        ax.errorbar(x_absorbed[linear_mask], alpha_array[linear_mask],
                   yerr=alpha_err_array[linear_mask],
                   fmt='o', color='#00AA00', markersize=8, capsize=4,
                   label='Data', zorder=3)

    # Add regression line
    slope = regression_stats['slope']
    intercept = regression_stats['intercept']

    x_line = np.linspace(0, max(x_absorbed) * 1.1, 100)
    y_line = slope * x_line + intercept

    if abs(intercept) < 1e-10:
        line_label = f'y = {slope:.2e}x'
    else:
        line_label = f'y = {slope:.2e}x + {intercept:.1f}'

    ax.plot(x_line, y_line, '--', color='black', linewidth=2, label=line_label, zorder=2)

    # Labels and formatting
    ax.set_xlabel('1 - 10$^{-A(λ)}$ (Absorbed Fraction)', fontsize=12)
    ax.set_ylabel('α (Amplitude, counts)', fontsize=12)
    ax.set_title('Beer-Lambert Linearity Check', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # No statistics text box (removed per user request)

    fig.tight_layout()

    # OPTIMIZED Phase 3: Lazy export data builder
    def _build_export_data():
        """Lazy builder for export data - only called when CSV export is requested."""
        return {
            'x_absorbed_fraction': x_absorbed,
            'alpha': alpha_array,
            'alpha_error': alpha_err_array,
            'absorbance_A_lambda': abs_array,
            'compound_labels': labels,
            'regression_slope': slope,
            'regression_intercept': intercept,
            'r_squared': regression_stats.get('r_squared', np.nan),
            'regression_line_x': x_line,
            'regression_line_y': y_line
        }

    fig.solis_export_data = _build_export_data

    logger.info(f"α vs A(λ) matplotlib plot created: slope={slope:.3e}")
    return fig, regression_stats


def plot_alpha_vs_intensity_mpl(
    selected_items: List[Dict[str, Any]],
    ei_unit: str = "a.u."
) -> Tuple[Figure, Dict[str, Any]]:
    """
    Create α vs Excitation Intensity correlation plot with linear regression using matplotlib.

    Parameters
    ----------
    selected_items : list of dict
        List of selected items from data browser
    ei_unit : str
        Unit string for excitation intensity (e.g., "mW", "μW")

    Returns
    -------
    tuple
        (fig, regression_stats): Matplotlib figure and regression statistics dict
    """
    # Extract data
    ei_values = []
    alpha_values = []
    alpha_errors = []
    labels = []

    for item in selected_items:
        ei_str = item.get('excitation_intensity', '')
        if ei_str == '—' or not ei_str:
            continue

        try:
            # Extract numeric part if string contains units (e.g., "50 mW" -> 50)
            ei_value = float(ei_str.split()[0]) if ' ' in ei_str else float(ei_str)
        except (ValueError, IndexError):
            continue

        analysis_result = item.get('analysis_result')
        if analysis_result is None:
            continue

        try:
            if isinstance(analysis_result, list):
                A_vals = [r.parameters.A for r in analysis_result if hasattr(r, 'parameters')]
                if A_vals:
                    alpha_mean = np.mean(A_vals)
                    alpha_sd = np.std(A_vals, ddof=1) if len(A_vals) > 1 else 0
                else:
                    continue
            elif isinstance(analysis_result, dict):
                if 'statistics' in analysis_result:
                    stats = analysis_result['statistics']
                    alpha_mean = stats.get('A_mean')
                    alpha_sd = stats.get('A_sd', 0)
                elif 'results' in analysis_result:
                    results_list = analysis_result['results']
                    A_vals = [r.parameters.A for r in results_list if hasattr(r, 'parameters')]
                    if A_vals:
                        alpha_mean = np.mean(A_vals)
                        alpha_sd = np.std(A_vals, ddof=1) if len(A_vals) > 1 else 0
                    else:
                        continue
                else:
                    continue

                if alpha_mean is None or alpha_mean == 'ND':
                    continue
            else:
                if hasattr(analysis_result, 'parameters'):
                    alpha_mean = analysis_result.parameters.A
                    alpha_sd = 0
                else:
                    continue

            ei_values.append(ei_value)
            alpha_values.append(alpha_mean)
            alpha_errors.append(alpha_sd)
            compound = item.get('compound', 'Unknown')
            labels.append(compound)

        except Exception as e:
            logger.warning(f"Error extracting data: {e}")
            continue

    if len(ei_values) < 2:
        logger.error(f"Insufficient EI data: found {len(ei_values)} points, need at least 2")
        logger.error(f"EI values found: {ei_values}")
        logger.error(f"Total items: {len(selected_items)}, items with EI: {sum(1 for item in selected_items if item.get('excitation_intensity'))}")
        raise ValueError("Need at least 2 data points with EI and α values")

    ei_array = np.array(ei_values)
    alpha_array = np.array(alpha_values)
    alpha_err_array = np.array(alpha_errors)

    # Perform linear regression
    regression_stats = perform_linear_regression_through_origin(ei_array, alpha_array)

    if not regression_stats['success']:
        raise ValueError("Linear regression failed")

    # Create matplotlib figure
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)

    # Plot all data points (no linear/saturation distinction)
    linear_mask = regression_stats['linear_mask']

    # Plot all points with same style
    if np.any(linear_mask):
        ax.errorbar(ei_array[linear_mask], alpha_array[linear_mask],
                   yerr=alpha_err_array[linear_mask],
                   fmt='o', color='#00AA00', markersize=8, capsize=4,
                   label='Data', zorder=3)

    # Add regression line
    slope = regression_stats['slope']
    intercept = regression_stats['intercept']

    x_line = np.linspace(0, max(ei_array) * 1.1, 100)
    y_line = slope * x_line + intercept

    if abs(intercept) < 1e-10:
        line_label = f'y = {slope:.2e}x'
    else:
        line_label = f'y = {slope:.2e}x + {intercept:.1f}'

    ax.plot(x_line, y_line, '--', color='black', linewidth=2, label=line_label, zorder=2)

    # Labels and formatting
    ax.set_xlabel(f'Excitation Intensity ({ei_unit})', fontsize=12)
    ax.set_ylabel('α (Amplitude, counts)', fontsize=12)
    ax.set_title('Intensity Dependence Check', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # No statistics text box (removed per user request)

    fig.tight_layout()

    # OPTIMIZED Phase 3: Lazy export data builder
    def _build_export_data():
        """Lazy builder for export data - only called when CSV export is requested."""
        return {
            'excitation_intensity': ei_array,
            'alpha': alpha_array,
            'alpha_error': alpha_err_array,
            'compound_labels': labels,
            'ei_unit': ei_unit,
            'regression_slope': slope,
            'regression_intercept': intercept,
            'r_squared': regression_stats.get('r_squared', np.nan),
            'regression_line_x': x_line,
            'regression_line_y': y_line
        }

    fig.solis_export_data = _build_export_data

    logger.info(f"α vs EI matplotlib plot created: slope={slope:.3e}")
    return fig, regression_stats
