#!/usr/bin/env python3
"""
Plot Viewer Widget for SOLIS

Floating/dockable widget for displaying interactive matplotlib figures using FigureCanvas.
Supports two modes:
1. Preview Mode: Raw data with spike detection and manual correction
2. Post-Analysis Mode: Complete fit results with residuals

Session 14: Migrated from Plotly/QWebEngineView to matplotlib for reliability
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox,
    QLabel, QToolBar, QMessageBox, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QAction
from pathlib import Path
import numpy as np

# Matplotlib imports
import matplotlib
matplotlib.use('QtAgg')  # Use Qt5/Qt6 backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from utils.logger_config import get_logger

logger = get_logger(__name__)


class PlotViewerWidget(QWidget):
    """Widget to display matplotlib figures with native Qt integration."""

    # Signals
    mask_parameters_changed = pyqtSignal(dict)  # Emitted when user changes mask (preview mode)
    window_closed = pyqtSignal()  # Emitted when window closes

    def __init__(self, parent=None, plot_title="Plot Viewer"):
        super().__init__(parent)

        # Store current data for re-plotting
        self.current_data = None
        self.current_fig = None  # matplotlib Figure object
        self.canvas = None  # FigureCanvas widget
        self.auto_mask_time = None  # Store original auto-detected mask
        self.plot_title = plot_title
        self.current_log_x = True  # Track current axis scale (default to log for decay plots)

        # Setup window flags for floating/dockable
        self.setWindowTitle(plot_title)
        self.setWindowFlags(Qt.WindowType.Window)  # Floating window
        self.resize(900, 700)

        # Setup UI
        self._setup_ui()

        # Show welcome message
        self._show_welcome_message()

    def _setup_ui(self):
        """Setup layout with toolbar, matplotlib canvas and mask controls."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # === Toolbar ===
        toolbar = QToolBar()
        toolbar.setMovable(False)

        # Toggle Log/Linear scale
        self.toggle_scale_action = QAction("Linear X-axis", self)
        self.toggle_scale_action.setCheckable(True)
        self.toggle_scale_action.setChecked(True)  # Start in log mode (default for decay plots)
        self.toggle_scale_action.setToolTip("Toggle between linear and logarithmic X-axis")
        self.toggle_scale_action.triggered.connect(self._toggle_scale)
        toolbar.addAction(self.toggle_scale_action)

        toolbar.addSeparator()

        # Export PDF
        export_pdf_action = QAction("Export PDF", self)
        export_pdf_action.setToolTip("Export plot as vector PDF")
        export_pdf_action.triggered.connect(self._export_pdf)
        toolbar.addAction(export_pdf_action)

        # Export CSV
        export_csv_action = QAction("Export CSV", self)
        export_csv_action.setToolTip("Export plot data as CSV")
        export_csv_action.triggered.connect(self._export_csv)
        toolbar.addAction(export_csv_action)

        layout.addWidget(toolbar)

        # === Matplotlib Navigation Toolbar (zoom, pan, home, etc.) ===
        # This will be created when the canvas is created
        self._mpl_toolbar = None

        # === Matplotlib canvas placeholder ===
        # Create a placeholder label that will be replaced with FigureCanvas
        self._canvas_placeholder = QLabel("Loading plot viewer...")
        self._canvas_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas_placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._canvas_placeholder, stretch=1)

        # === Mask control panel (initially hidden) ===
        self.mask_panel = QWidget()
        self.mask_panel.setMaximumHeight(50)
        mask_layout = QHBoxLayout(self.mask_panel)
        mask_layout.setContentsMargins(10, 5, 10, 5)

        # Label
        mask_layout.addWidget(QLabel("Spike Mask Endpoint (μs):"))

        # Spinbox for mask time
        self.mask_spinbox = QDoubleSpinBox()
        self.mask_spinbox.setDecimals(4)
        self.mask_spinbox.setMinimum(0.0001)
        self.mask_spinbox.setMaximum(1000.0)
        self.mask_spinbox.setSingleStep(0.01)
        self.mask_spinbox.setValue(0.1)
        self.mask_spinbox.setMinimumWidth(120)
        # Connect valueChanged to update preview dynamically
        self.mask_spinbox.valueChanged.connect(self._on_mask_preview_update)
        mask_layout.addWidget(self.mask_spinbox)

        # Apply button
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply_mask)
        self.apply_btn.setMaximumWidth(80)
        self.apply_btn.setToolTip("Apply new mask endpoint to current replicate")
        mask_layout.addWidget(self.apply_btn)

        # Apply to All button
        self.apply_all_btn = QPushButton("Apply to All")
        self.apply_all_btn.clicked.connect(self._on_apply_to_all)
        self.apply_all_btn.setMaximumWidth(100)
        self.apply_all_btn.setToolTip("Apply this mask endpoint to ALL replicates")
        mask_layout.addWidget(self.apply_all_btn)

        # Reset button
        self.reset_btn = QPushButton("Reset to Auto")
        self.reset_btn.clicked.connect(self._on_reset_mask)
        self.reset_btn.setMaximumWidth(100)
        self.reset_btn.setToolTip("Reset to automatic spike detection")
        mask_layout.addWidget(self.reset_btn)

        # Hide mask panel initially
        self.mask_panel.hide()

        layout.addWidget(self.mask_panel, stretch=0)

    def showEvent(self, event):
        """Handle widget show event - regenerate plot if needed after reopen."""
        super().showEvent(event)

        logger.debug(f"showEvent triggered for {self.plot_title}")
        logger.debug(f"  current_fig: {self.current_fig is not None}")
        logger.debug(f"  canvas: {self.canvas is not None}")

        # If we have figure data but no canvas, regenerate
        if self.current_fig is not None and self.canvas is None:
            logger.info(f"Regenerating plot on reopen: {self.plot_title}")
            self.show_matplotlib_figure(self.current_fig)
        elif self.current_data is not None and self.canvas is None:
            logger.info(f"Regenerating plot from data on reopen: {self.plot_title}")
            self._render_plot(self.current_data)

    def closeEvent(self, event):
        """Handle widget close event - keep figure for reopening."""
        try:
            # Keep self.current_fig and self.current_data for menu reopen functionality
            logger.debug(f"closeEvent for {self.plot_title} - preserving data")
        except Exception as e:
            logger.error(f"Error during closeEvent: {e}")

        self.window_closed.emit()
        event.accept()

    def destroy_plot(self):
        """Completely destroy all plot data and resources - called on session reset."""
        logger.info(f"Destroying plot: {self.plot_title}")
        try:
            # Clear matplotlib figure
            if self.current_fig is not None:
                plt.close(self.current_fig)
                self.current_fig = None

            # Clear canvas
            if self.canvas is not None:
                self.canvas.deleteLater()
                self.canvas = None

            # Clear data
            self.current_data = None
            logger.debug("Cleared figure and data references")
        except Exception as e:
            logger.error(f"Error in destroy_plot: {e}")

    def _toggle_scale(self, checked):
        """Toggle between log and linear X-axis - works for ALL plot types."""
        if not self.current_fig:
            return

        # Check CURRENT state of first axis to determine what to do
        axes = self.current_fig.get_axes()
        if not axes:
            return

        current_scale = axes[0].get_xscale()

        # Toggle based on current axis state (not button state)
        if current_scale == 'linear':
            # Switch to log
            new_scale = 'log'
            self.current_log_x = True
            self.toggle_scale_action.setText("Linear X-axis")
            self.toggle_scale_action.setChecked(True)
            for ax in axes:
                ax.set_xscale('log')
                xlim = ax.get_xlim()
                ax.set_xlim(max(0.01, xlim[0]), xlim[1])  # Prevent log(0)
        else:
            # Switch to linear
            new_scale = 'linear'
            self.current_log_x = False
            self.toggle_scale_action.setText("Log X-axis")
            self.toggle_scale_action.setChecked(False)
            for ax in axes:
                ax.set_xscale('linear')
                xlim = ax.get_xlim()
                ax.set_xlim(0.0, xlim[1])  # Start from 0

        # Redraw canvas
        if self.canvas:
            self.canvas.draw_idle()  # More efficient than draw()
            logger.debug(f"Toggle scale: {current_scale} → {new_scale}")

    def _export_pdf(self):
        """Export current plot as vector PDF."""
        if not self.current_fig:
            QMessageBox.warning(self, "No Plot", "No plot available to export.")
            return

        # Sanitize filename: replace invalid characters (Windows doesn't allow : < > " / \ | ? *)
        safe_title = self.plot_title.replace(":", "_").replace("<", "_").replace(">", "_").replace('"', "_").replace("/", "_").replace("\\", "_").replace("|", "_").replace("?", "_").replace("*", "_")

        # Get filename from user
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot as PDF",
            f"{safe_title}.pdf",
            "PDF Files (*.pdf)"
        )

        if filename:
            try:
                # Use matplotlib's native PDF export (vector format)
                self.current_fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
                logger.info(f"Plot exported to PDF: {filename}")
                QMessageBox.information(self, "Export Complete", f"Plot exported to:\n{filename}")
            except Exception as e:
                logger.error(f"PDF export failed: {e}")
                QMessageBox.critical(self, "Export Error", f"Failed to export PDF:\n{str(e)}")

    def _export_csv(self):
        """Export plot data as CSV."""
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "No plot data available to export.")
            return

        # Sanitize filename: replace invalid characters (Windows doesn't allow : < > " / \ | ? *)
        safe_title = self.plot_title.replace(":", "_").replace("<", "_").replace(">", "_").replace('"', "_").replace("/", "_").replace("\\", "_").replace("|", "_").replace("?", "_").replace("*", "_")

        # Get filename from user
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot Data as CSV",
            f"{safe_title}_data.csv",
            "CSV Files (*.csv)"
        )

        if filename:
            try:
                import pandas as pd

                # === Detect plot type and prepare data ===
                data_dict = {}

                # 1a. HETEROGENEOUS CHI-SQUARE LANDSCAPE (from heterogeneous_plotter_new.py)
                if 'chi2_surface' in self.current_data and 'tau_T_grid' in self.current_data:
                    # Export 2D chi-square surface as CSV with tau_T rows and tau_delta_W columns
                    tau_T_grid = self.current_data['tau_T_grid']
                    tau_delta_W_grid = self.current_data['tau_delta_W_grid']
                    chi2_surface = self.current_data['chi2_surface']

                    # Create DataFrame with tau_T as index and tau_delta_W as columns
                    df = pd.DataFrame(chi2_surface, index=tau_T_grid, columns=tau_delta_W_grid)
                    df.index.name = 'tau_T_us'
                    df.columns.name = 'tau_delta_W_us'

                    df.to_csv(filename)
                    logger.info(f"Chi-square landscape exported to CSV: {filename}")
                    QMessageBox.information(self, "Export Complete", f"Chi-square landscape exported to:\n{filename}\n\nRows: tau_T values\nColumns: tau_delta_W values\nCells: chi^2_reduced")
                    return  # Early return, already saved

                # 1b. HETEROGENEOUS GRID PLOT (chi-square landscape - old format from solis_plotter.py)
                elif 'grid_dataframe' in self.current_data:
                    df = self.current_data['grid_dataframe']
                    df.to_csv(filename, index=False)
                    logger.info(f"Grid data exported to CSV: {filename}")
                    QMessageBox.information(self, "Export Complete", f"Grid search data exported to:\n{filename}")
                    return  # Early return, already saved

                # 2a. HETEROGENEOUS FIT CURVES (from heterogeneous_plotter_new.py)
                elif 'time_experimental' in self.current_data and 'intensity_experimental' in self.current_data and 'fit_mask' in self.current_data:
                    time_exp = self.current_data['time_experimental']
                    data_dict['Time_us'] = time_exp
                    data_dict['Intensity_Experimental'] = self.current_data['intensity_experimental']
                    data_dict['Intensity_Fitted'] = self.current_data['intensity_fitted']
                    data_dict['Residuals'] = self.current_data['residuals']

                    # Weighted residuals may have different length (only fitted region)
                    weighted_res = self.current_data['weighted_residuals']
                    fit_mask = self.current_data['fit_mask']
                    if len(weighted_res) != len(time_exp):
                        # Pad with NaN to match full time array
                        full_weighted_res = np.full(len(time_exp), np.nan)
                        if len(weighted_res) == np.sum(fit_mask):
                            full_weighted_res[fit_mask] = weighted_res
                        else:
                            # Fallback: put at end
                            full_weighted_res[-len(weighted_res):] = weighted_res
                        data_dict['Weighted_Residuals'] = full_weighted_res
                    else:
                        data_dict['Weighted_Residuals'] = weighted_res

                    data_dict['Fit_Mask'] = fit_mask.astype(int)

                    # Add components if available
                    if self.current_data.get('lipid_component') is not None:
                        data_dict['Lipid_Component'] = self.current_data['lipid_component']
                    if self.current_data.get('water_component') is not None:
                        data_dict['Water_Component'] = self.current_data['water_component']

                # 2b. HETEROGENEOUS FIT PLOT (old format from solis_plotter.py - with lipid/water components)
                elif 'signal_exp' in self.current_data and 'lipid_component' in self.current_data:
                    data_dict['Time_us'] = self.current_data['time_us']
                    data_dict['Signal_Exp'] = self.current_data['signal_exp']
                    data_dict['Signal_Fit'] = self.current_data['signal_fit']
                    data_dict['Weighted_Residuals'] = self.current_data['weighted_residuals']
                    if not np.all(np.isnan(self.current_data.get('lipid_component', []))):
                        data_dict['Lipid_Component'] = self.current_data['lipid_component']
                        data_dict['Water_Component'] = self.current_data['water_component']

                # 3. SURPLUS ANALYSIS PLOT (4-panel plot with all steps)
                elif 'surplus_signal' in self.current_data and 'late_fit_curve' in self.current_data:
                    time = self.current_data['time_us']
                    data_dict['Time_us'] = time
                    data_dict['Intensity_Raw'] = self.current_data['intensity_raw']
                    data_dict['Late_Fit_Curve'] = self.current_data['late_fit_curve']
                    data_dict['Surplus_Signal'] = self.current_data['surplus_signal']
                    data_dict['Surplus_Fit_Curve'] = self.current_data['surplus_fit_curve']
                    data_dict['Final_Heterogeneous_Fit'] = self.current_data['final_curve']

                    # Weighted residuals may have different length (only valid points)
                    # Pad with NaN to match time array length
                    weighted_res = self.current_data['weighted_residuals']
                    if len(weighted_res) != len(time):
                        # Create full-length array filled with NaN
                        full_weighted_res = np.full(len(time), np.nan)
                        # Find valid mask and fill in the valid residuals
                        if 'fitting_mask' in self.current_data:
                            valid_mask = self.current_data['fitting_mask']
                            intensity_raw = self.current_data['intensity_raw']
                            valid_indices = np.isfinite(intensity_raw)
                            # Only fill where both masks are true
                            combined_mask = valid_indices
                            if len(weighted_res) == np.sum(combined_mask):
                                full_weighted_res[combined_mask] = weighted_res
                            else:
                                # If lengths still don't match, just put residuals at end
                                full_weighted_res[-len(weighted_res):] = weighted_res
                        else:
                            # No mask info, just append to end
                            full_weighted_res[-len(weighted_res):] = weighted_res
                        data_dict['Weighted_Residuals'] = full_weighted_res
                    else:
                        data_dict['Weighted_Residuals'] = weighted_res

                    if 'fitting_mask' in self.current_data:
                        data_dict['Fitting_Mask'] = self.current_data['fitting_mask'].astype(int)

                # 4. MEAN DECAY PLOT (mean ± SD)
                elif 'mean_intensity_raw' in self.current_data:
                    data_dict['Time_us'] = self.current_data['time_us']
                    data_dict['Mean_Intensity_Raw'] = self.current_data['mean_intensity_raw']
                    data_dict['SD_Intensity_Raw'] = self.current_data['sd_intensity_raw']
                    data_dict['Mean_Main_Curve'] = self.current_data['mean_main_curve']
                    data_dict['SD_Main_Curve'] = self.current_data['sd_main_curve']
                    data_dict['Mean_Weighted_Residuals'] = self.current_data['mean_weighted_residuals']
                    data_dict['SD_Weighted_Residuals'] = self.current_data['sd_weighted_residuals']
                    if self.current_data.get('fitting_mask') is not None:
                        data_dict['Fitting_Mask'] = self.current_data['fitting_mask'].astype(int)

                # 5. BATCH SUMMARY PLOT (multiple replicates)
                elif 'n_replicates' in self.current_data:
                    n_reps = self.current_data['n_replicates']
                    # Export each replicate's data
                    for i in range(1, n_reps + 1):
                        if f'rep{i}_time_us' in self.current_data:
                            data_dict[f'Rep{i}_Time_us'] = self.current_data[f'rep{i}_time_us']
                            data_dict[f'Rep{i}_Intensity_Raw'] = self.current_data[f'rep{i}_intensity_raw']
                            data_dict[f'Rep{i}_Main_Curve'] = self.current_data[f'rep{i}_main_curve']
                            data_dict[f'Rep{i}_Weighted_Residuals'] = self.current_data[f'rep{i}_main_weighted_residuals']
                            if f'rep{i}_fitting_mask' in self.current_data:
                                data_dict[f'Rep{i}_Fitting_Mask'] = self.current_data[f'rep{i}_fitting_mask'].astype(int)
                    # Add mean and std if available
                    if 'mean_curve' in self.current_data:
                        data_dict['Time_us'] = self.current_data['time_us']
                        data_dict['Mean_Curve'] = self.current_data['mean_curve']
                        data_dict['Std_Curve'] = self.current_data['std_curve']

                # 6. MERGED DECAY PLOT (multiple compounds/datasets)
                elif 'datasets' in self.current_data and 'n_datasets' in self.current_data:
                    # Export each dataset separately into columns
                    for idx, dataset in enumerate(self.current_data['datasets'], 1):
                        compound = dataset['compound']
                        prefix = f"{compound}_Dataset{idx}"

                        data_dict[f'{prefix}_Time_us'] = dataset['time_us']

                        if dataset['type'] == 'mean':
                            data_dict[f'{prefix}_Mean_Intensity'] = dataset['mean_intensity']
                            data_dict[f'{prefix}_SD_Intensity'] = dataset['sd_intensity']
                            data_dict[f'{prefix}_Mean_Fit'] = dataset['mean_fit']
                            data_dict[f'{prefix}_SD_Fit'] = dataset['sd_fit']
                            data_dict[f'{prefix}_Mean_Residuals'] = dataset['mean_residuals']
                            data_dict[f'{prefix}_SD_Residuals'] = dataset['sd_residuals']
                        else:  # replicate
                            rep_num = dataset.get('replicate_num', idx)
                            data_dict[f'{prefix}_Rep{rep_num}_Intensity'] = dataset['intensity']
                            data_dict[f'{prefix}_Rep{rep_num}_Fit'] = dataset['fit']
                            data_dict[f'{prefix}_Rep{rep_num}_Residuals'] = dataset['residuals']

                # 7. SINGLE DECAY PLOT (standard homogeneous fit)
                elif 'intensity_raw' in self.current_data and 'main_curve' in self.current_data:
                    data_dict['Time_us'] = self.current_data['time_us']
                    data_dict['Intensity_Raw'] = self.current_data['intensity_raw']
                    data_dict['Main_Curve'] = self.current_data['main_curve']
                    data_dict['Main_Weighted_Residuals'] = self.current_data['main_weighted_residuals']

                    # Add literature curve if available
                    if 'literature_curve' in self.current_data and self.current_data['literature_curve'] is not None:
                        data_dict['Literature_Curve'] = self.current_data['literature_curve']
                        data_dict['Literature_Weighted_Residuals'] = self.current_data['literature_weighted_residuals']

                    # Add masks if available
                    if self.current_data.get('fitting_mask') is not None:
                        data_dict['Fitting_Mask'] = self.current_data['fitting_mask'].astype(int)
                    if self.current_data.get('spike_region') is not None:
                        data_dict['Spike_Region'] = self.current_data['spike_region'].astype(int)

                # 8. LINEARITY PLOT: α vs Absorption (Beer-Lambert check)
                elif 'x_absorbed_fraction' in self.current_data and 'alpha' in self.current_data:
                    # Export data points separately from regression line (different lengths)
                    # First export the actual data points
                    data_dict['Absorbed_Fraction'] = self.current_data['x_absorbed_fraction']
                    data_dict['Alpha'] = self.current_data['alpha']
                    data_dict['Alpha_Error'] = self.current_data['alpha_error']
                    if 'absorbance_A_lambda' in self.current_data:
                        data_dict['Absorbance_A_lambda'] = self.current_data['absorbance_A_lambda']
                    if 'compound_labels' in self.current_data:
                        data_dict['Compound'] = self.current_data['compound_labels']

                    # Pad regression line arrays to match data point count
                    # (regression line has ~100 points, data has N points)
                    n_data = len(self.current_data['x_absorbed_fraction'])
                    reg_x = self.current_data['regression_line_x']
                    reg_y = self.current_data['regression_line_y']

                    # Pad regression arrays with NaN to match data length
                    if len(reg_x) > n_data:
                        # Regression line is longer - truncate or create separate columns
                        # Better approach: create separate rows with NaN for data columns
                        padded_reg_x = np.full(n_data, np.nan)
                        padded_reg_y = np.full(n_data, np.nan)
                        data_dict['Regression_Line_X'] = padded_reg_x
                        data_dict['Regression_Line_Y'] = padded_reg_y
                        # Save full regression line in separate DataFrame
                        regression_df = pd.DataFrame({
                            'Regression_Line_X': reg_x,
                            'Regression_Line_Y': reg_y
                        })
                        # We'll append this after the main data
                        data_dict['__regression_df__'] = regression_df
                    else:
                        # Regression line is shorter or equal - just pad with NaN
                        padded_reg_x = np.full(n_data, np.nan)
                        padded_reg_y = np.full(n_data, np.nan)
                        padded_reg_x[:len(reg_x)] = reg_x
                        padded_reg_y[:len(reg_y)] = reg_y
                        data_dict['Regression_Line_X'] = padded_reg_x
                        data_dict['Regression_Line_Y'] = padded_reg_y

                # 9. LINEARITY PLOT: α vs Excitation Intensity
                elif 'excitation_intensity' in self.current_data and 'alpha' in self.current_data:
                    ei_unit = self.current_data.get('ei_unit', 'a.u.')
                    # Export data points
                    data_dict[f'Excitation_Intensity_{ei_unit}'] = self.current_data['excitation_intensity']
                    data_dict['Alpha'] = self.current_data['alpha']
                    data_dict['Alpha_Error'] = self.current_data['alpha_error']
                    if 'compound_labels' in self.current_data:
                        data_dict['Compound'] = self.current_data['compound_labels']

                    # Pad regression line arrays to match data point count
                    n_data = len(self.current_data['excitation_intensity'])
                    reg_x = self.current_data['regression_line_x']
                    reg_y = self.current_data['regression_line_y']

                    # Pad regression arrays with NaN to match data length
                    if len(reg_x) > n_data:
                        # Regression line is longer - create separate section
                        padded_reg_x = np.full(n_data, np.nan)
                        padded_reg_y = np.full(n_data, np.nan)
                        data_dict['Regression_Line_X'] = padded_reg_x
                        data_dict['Regression_Line_Y'] = padded_reg_y
                        # Save full regression line in separate DataFrame
                        regression_df = pd.DataFrame({
                            'Regression_Line_X': reg_x,
                            'Regression_Line_Y': reg_y
                        })
                        data_dict['__regression_df__'] = regression_df
                    else:
                        # Regression line is shorter or equal - just pad with NaN
                        padded_reg_x = np.full(n_data, np.nan)
                        padded_reg_y = np.full(n_data, np.nan)
                        padded_reg_x[:len(reg_x)] = reg_x
                        padded_reg_y[:len(reg_y)] = reg_y
                        data_dict['Regression_Line_X'] = padded_reg_x
                        data_dict['Regression_Line_Y'] = padded_reg_y

                # 10. PREVIEW MODE (raw data with spike detection)
                elif self.current_data.get('preview_mode'):
                    time = self.current_data.get('time')
                    intensity = self.current_data.get('intensity')
                    spike_mask = self.current_data.get('spike_mask')

                    data_dict['Time_us'] = time
                    data_dict['Intensity'] = intensity
                    if spike_mask is not None:
                        data_dict['Is_Signal'] = spike_mask.astype(int)  # 1=signal, 0=spike

                # 11. FALLBACK: Old format with result object
                elif 'result' in self.current_data:
                    result = self.current_data['result']
                    data_dict['Time_us'] = result.time_experiment_us
                    data_dict['Intensity_Raw'] = result.intensity_raw
                    data_dict['Main_Fit'] = result.main_curve
                    data_dict['Weighted_Residuals'] = result.main_weighted_residuals

                    if result.literature.success and result.literature.curve is not None:
                        data_dict['Literature_Fit'] = result.literature.curve
                        data_dict['Literature_Residuals'] = result.literature.weighted_residuals

                else:
                    raise ValueError("Unknown plot data format - cannot determine export structure")

                # Check if we have a separate regression DataFrame (for linearity plots)
                regression_df = data_dict.pop('__regression_df__', None)

                # Save to CSV
                df = pd.DataFrame(data_dict)

                if regression_df is not None:
                    # For linearity plots with regression line:
                    # Append regression line data as separate rows below the main data
                    # Add empty rows separator for clarity
                    separator = pd.DataFrame({col: [''] for col in df.columns})
                    header_row = pd.DataFrame({col: ['===REGRESSION LINE==='] if i == 0 else ['']
                                               for i, col in enumerate(df.columns)})

                    # Combine: data + blank row + header + regression line
                    # Need to align columns properly
                    regression_df_aligned = regression_df.reindex(columns=df.columns, fill_value='')
                    combined_df = pd.concat([df, separator, header_row, regression_df_aligned], ignore_index=True)
                    combined_df.to_csv(filename, index=False)
                    logger.info(f"Plot data with regression line exported to CSV: {filename}")
                    QMessageBox.information(self, "Export Complete",
                                           f"Data exported to:\n{filename}\n\n"
                                           f"Data points: {len(df)} rows\n"
                                           f"Regression line: {len(regression_df)} points (appended below)")
                else:
                    # Standard export
                    df.to_csv(filename, index=False)
                    logger.info(f"Plot data exported to CSV: {filename}")
                    QMessageBox.information(self, "Export Complete", f"Data exported to:\n{filename}")

            except Exception as e:
                logger.error(f"CSV export failed: {e}")
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{str(e)}")

    def _show_welcome_message(self):
        """Display welcome message in plot viewer."""
        # Simply keep the placeholder label visible
        # The label already says "Loading plot viewer..."
        self.mask_panel.hide()  # Hide mask controls when showing welcome

    def _on_mask_preview_update(self, value):
        """Update plot preview when mask spinbox value changes (without saving)."""
        if self.current_data is None or not self.current_data.get('preview_mode', False):
            return

        # Update plot with new mask position (visual preview only, not saved)
        new_mask_time = value

        # Create updated data with new mask
        updated_data = self.current_data.copy()
        updated_data['mask_end_time'] = new_mask_time

        # Recreate mask based on new endpoint
        time = updated_data.get('time')
        if time is not None:
            # Find closest index to new mask time
            mask_idx = np.searchsorted(time, new_mask_time)
            updated_data['spike_mask'] = np.arange(len(time)) > mask_idx

        # Update current_data to reflect the new mask (but don't save to storage)
        # This allows the next spinbox change to work from the current state
        self.current_data = updated_data

        # Re-plot with new mask (visual update only)
        self._render_plot(updated_data)

    def _on_apply_mask(self):
        """Handle Apply button click - update mask for CURRENT replicate only."""
        if self.current_data is None:
            return

        new_mask_time = self.mask_spinbox.value()

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Apply Mask Correction",
            f"Apply new mask endpoint ({new_mask_time:.4f} μs) to current replicate only?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._apply_mask_correction(new_mask_time, apply_to_all=False)

    def _on_apply_to_all(self):
        """Handle Apply to All button click - update mask for ALL replicates."""
        if self.current_data is None:
            return

        new_mask_time = self.mask_spinbox.value()

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Apply Mask to All",
            f"Apply new mask endpoint ({new_mask_time:.4f} μs) to ALL replicates in this compound?\n\n"
            "This will override automatic spike detection for all replicates.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._apply_mask_correction(new_mask_time, apply_to_all=True)

    def _apply_mask_correction(self, new_mask_time: float, apply_to_all: bool):
        """
        Apply mask correction and emit signal.

        Parameters
        ----------
        new_mask_time : float
            New mask endpoint in microseconds
        apply_to_all : bool
            If True, apply to all replicates; if False, only current
        """
        # Update current data with new mask
        updated_data = self.current_data.copy()
        updated_data['mask_end_time'] = new_mask_time

        # Recreate mask based on new endpoint
        time = updated_data.get('time')
        if time is not None:
            # Find closest index to new mask time
            mask_idx = np.searchsorted(time, new_mask_time)
            updated_data['spike_mask'] = np.arange(len(time)) > mask_idx

        # Re-plot with new mask
        self._render_plot(updated_data)

        # Emit signal with correction parameters
        self.mask_parameters_changed.emit({
            'mask_end_time': new_mask_time,
            'apply_to_all': apply_to_all,
            'compound_name': updated_data.get('compound_name'),
            'replicate_num': updated_data.get('replicate_num')
        })

        logger.info(f"Mask correction applied: {new_mask_time:.4f} μs, apply_to_all={apply_to_all}")

    def _on_reset_mask(self):
        """Handle Reset button click - restore auto-detected mask."""
        if self.current_data is None or self.auto_mask_time is None:
            return

        # Reset spinbox to auto value
        self.mask_spinbox.setValue(self.auto_mask_time)

        # Restore original mask
        updated_data = self.current_data.copy()
        updated_data['mask_end_time'] = self.auto_mask_time

        # Recreate original mask
        time = updated_data.get('time')
        if time is not None:
            mask_idx = np.searchsorted(time, self.auto_mask_time)
            updated_data['spike_mask'] = np.arange(len(time)) > mask_idx

        # Re-plot
        self._render_plot(updated_data)

        logger.info(f"Mask reset to automatic: {self.auto_mask_time:.4f} μs")

    def update_plot(self, data):
        """
        Update plot from data dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing plot data. Expected keys:
            - 'file_type': 'decay' or 'absorption'
            - 'result': KineticsResult object (for post-analysis)
            - 'time': time data array (if result is None)
            - 'intensity': intensity data array (if result is None)
            - 'wavelength': wavelength array (for absorption)
            - 'absorbance': absorbance array (for absorption)
            - 'log_x': bool, whether to use log scale for x-axis (optional)
            - 'file_path': path to the file being displayed
            - 'preview_mode': bool, whether in preview mode (for mask controls)
            - 'mask_end_time': float, spike mask endpoint time
            - 'compound_name': str, compound name (for tracking)
            - 'replicate_num': int, replicate number (for tracking)
        """
        if not data:
            return

        # Store current data for re-plotting
        self.current_data = data.copy()

        # Use current log_x state if not specified in data
        if 'log_x' not in data or data['log_x'] is None:
            data['log_x'] = self.current_log_x

        # Render the plot
        self._render_plot(data)

    def _render_plot(self, data):
        """
        Internal method to render plot from data.

        Parameters
        ----------
        data : dict
            Plot data dictionary
        """
        try:
            # Import here to avoid circular imports
            from plotting.solis_plotter import SOLISPlotter
            from matplotlib.figure import Figure

            # Extract data
            file_type = data.get('file_type', 'decay')
            file_path = data.get('file_path', 'Unknown')
            preview_mode = data.get('preview_mode', False)
            mask_end_time = data.get('mask_end_time')
            log_x = data.get('log_x', self.current_log_x)

            # Update toolbar state
            self.toggle_scale_action.setChecked(log_x)
            if log_x:
                self.toggle_scale_action.setText("Linear X-axis")
            else:
                self.toggle_scale_action.setText("Log X-axis")

            # Show/hide mask controls based on mode
            if preview_mode and mask_end_time is not None:
                # Store auto-detected mask time (when loading new file)
                # Check if this is a new file by comparing file paths
                if (self.current_data.get('file_path') != data.get('file_path') or
                    self.auto_mask_time is None):
                    self.auto_mask_time = mask_end_time

                # Update spinbox with current mask time (block signals to prevent recursion)
                self.mask_spinbox.blockSignals(True)
                self.mask_spinbox.setValue(mask_end_time)
                self.mask_spinbox.blockSignals(False)

                # Show mask controls
                self.mask_panel.show()
            else:
                # Hide mask controls for non-preview or post-analysis plots
                self.mask_panel.hide()
                self.auto_mask_time = None

            # Handle absorption spectra
            if file_type == 'absorption':
                wavelength = data.get('wavelength')
                absorbance = data.get('absorbance')

                if wavelength is None or absorbance is None:
                    return

                # Create absorption spectrum plot with matplotlib
                from matplotlib.figure import Figure
                fig = Figure(figsize=(7, 6), dpi=100)
                ax = fig.add_subplot(111)
                ax.plot(wavelength, absorbance, '-', color='#2c3e50', linewidth=2, label='Absorption')
                ax.set_xlabel('Wavelength (nm)', fontsize=12)
                ax.set_ylabel('Absorbance', fontsize=12)
                ax.set_title(f'Absorption Spectrum: {Path(file_path).stem}', fontsize=14)
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                self.show_matplotlib_figure(fig)
                return

            # Handle decay plots
            result = data.get('result')
            mean_arrays = data.get('mean_arrays')  # For mean plots
            sd_arrays = data.get('sd_arrays')
            preview_mode = data.get('preview_mode', False)

            # Create plotter
            plotter = SOLISPlotter()

            if mean_arrays is not None and sd_arrays is not None:
                # MEAN PLOT: Plot computed mean with SD envelopes
                compound_name = data.get('compound_name', 'Compound')
                title = f'{compound_name} - Mean ± SD'
                fig = plotter.plot_mean_decay_mpl(mean_arrays, sd_arrays, log_x=log_x, title=title)
            elif result is not None:
                # POST-ANALYSIS: We have a fit result - use full 2-panel plotting
                fig = plotter.plot_single_decay_mpl(result, log_x=log_x)
            elif preview_mode:
                # PREVIEW MODE: Raw data + shaded spike region + colored points
                time = data.get('time')
                intensity = data.get('intensity')
                spike_mask = data.get('spike_mask')
                mask_end_time = data.get('mask_end_time')
                snr_result = data.get('snr_result')
                display_label = data.get('display_label', '')

                if time is None or intensity is None:
                    return

                fig = self._create_preview_plot_mpl(
                    time, intensity, spike_mask, mask_end_time,
                    snr_result, file_path, log_x, display_label
                )
            else:
                # Fallback: No fit result - plot raw data only (matplotlib)
                time = data.get('time')
                intensity = data.get('intensity')

                if time is None or intensity is None:
                    return

                # Create simple scatter plot with matplotlib
                fig = Figure(figsize=(7, 6), dpi=100)
                ax = fig.add_subplot(111)
                ax.plot(time, intensity, 'o', markersize=4, alpha=0.7, color='#3498db', label='Raw Data')
                ax.set_xlabel('Time (μs)', fontsize=12)
                ax.set_ylabel('Intensity (a.u.)', fontsize=12)
                ax.set_title(f'Decay Data: {Path(file_path).stem}', fontsize=14)
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                if log_x:
                    ax.set_xscale('log')
                fig.tight_layout()

            # Display the figure
            self.show_matplotlib_figure(fig)

        except Exception as e:
            # Display error message
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"Plot rendering failed: {error_detail}")

            # Show error dialog
            QMessageBox.critical(
                self,
                "Plot Rendering Error",
                f"Failed to generate plot:\n\n{str(e)}\n\nSee console for full traceback."
            )

    def _create_preview_plot_mpl(self, time, intensity, spike_mask, mask_end_time,
                                  snr_result, file_path, log_x, display_label=''):
        """
        Create preview plot with raw data + shaded spike region using matplotlib.

        Shows:
        - Raw data points (colored: spike=gray, signal=black)
        - Shaded spike region (gray overlay)
        - Vertical line at mask endpoint
        - SNR info in title

        Parameters
        ----------
        time : array
            Time data
        intensity : array
            Intensity data
        spike_mask : array of bool
            Mask (True=signal, False=spike)
        mask_end_time : float
            End time of spike mask
        snr_result : SNRResult
            SNR analysis result
        file_path : str
            File path for title
        log_x : bool
            Use log scale for x-axis
        display_label : str
            Label to show (e.g., "Replicate 1")

        Returns
        -------
        matplotlib.figure.Figure
            Preview plot figure
        """
        from matplotlib.figure import Figure

        # Create figure
        fig = Figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Determine point colors based on mask
        if spike_mask is not None:
            # Spike points (gray)
            spike_indices = ~spike_mask
            if np.any(spike_indices):
                ax.plot(time[spike_indices], intensity[spike_indices], 'o',
                       color='#95a5a6', markersize=5, alpha=0.6, label='Spike Region')

            # Signal points (black)
            signal_indices = spike_mask
            if np.any(signal_indices):
                ax.plot(time[signal_indices], intensity[signal_indices], 'o',
                       color='#2c3e50', markersize=5, alpha=0.8, label='Signal')
        else:
            # No mask - show all points in black
            ax.plot(time, intensity, 'o', color='#2c3e50', markersize=5, alpha=0.8, label='Raw Data')

        # Add shaded spike region and vertical line at mask endpoint
        if mask_end_time is not None:
            y_max = np.max(intensity) if len(intensity) > 0 else 1000
            magnitude = 10 ** np.floor(np.log10(y_max))
            y_max_rounded = np.ceil(y_max / magnitude) * magnitude

            # Shaded rectangle from x=0 to mask_end_time
            ax.axvspan(time[0] if log_x else 0, mask_end_time, alpha=0.2, color='gray', zorder=0)

            # Vertical line at mask endpoint
            ax.axvline(x=mask_end_time, color='red', linewidth=2, linestyle='--',
                      label=f'Mask: {mask_end_time:.3f} μs')

        # Build title with SNR info and display label
        title_text = f'Preview: {Path(file_path).stem}'
        if display_label:
            title_text += f' [{display_label}]'
        if snr_result:
            snr_linear = snr_result.snr_linear
            snr_quality = snr_result.quality
            title_text += f'\nSNR: {snr_linear:.1f}:1 ({snr_quality})'

        ax.set_title(title_text, fontsize=12)
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Intensity (counts)', fontsize=12)

        # Set axis scales and ranges
        if log_x:
            ax.set_xscale('log')
            ax.set_xlim(0.01, 100)
        else:
            ax.set_xscale('linear')
            ax.set_xlim(0, 50)

        # Y-axis: Auto-scale with 0 at bottom
        ax.set_ylim(bottom=0, auto=True)
        ax.autoscale(enable=True, axis='y', tight=False)

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        return fig

    def show_matplotlib_figure(self, fig):
        """
        Display a matplotlib figure in the viewer.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to display
        """
        logger.info(f"show_matplotlib_figure called with fig type: {type(fig)}")
        self.current_fig = fig

        # Extract attached data for CSV export
        if hasattr(fig, 'solis_export_data'):
            # Result plot with export data - use it
            self.current_data = fig.solis_export_data
            logger.info("Extracted export data from figure")
        else:
            # No export data - could be preview plot or result plot without data
            # If current_data is from a different plot (different file_path), clear it
            # Otherwise keep it (for preview plots that set current_data via update_plot)
            if self.current_data and self.current_data.get('preview_mode'):
                # It's preview data - keep it for preview mode functionality
                logger.debug("Keeping preview mode current_data")
            else:
                # Not preview data or no current_data - clear it so toggle uses axes directly
                self.current_data = None
                logger.debug("Clearing current_data (no solis_export_data on figure)")

        try:
            # Remove old canvas if it exists
            if self.canvas is not None:
                layout = self.layout()
                layout.removeWidget(self.canvas)
                self.canvas.deleteLater()
                self.canvas = None

            # Create new canvas from the figure
            self.canvas = FigureCanvas(fig)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            # Create matplotlib navigation toolbar (zoom, pan, home, save)
            if self._mpl_toolbar is not None:
                layout = self.layout()
                layout.removeWidget(self._mpl_toolbar)
                self._mpl_toolbar.deleteLater()

            self._mpl_toolbar = NavigationToolbar(self.canvas, self)

            # Replace placeholder with toolbar + canvas
            layout = self.layout()
            if hasattr(self, '_canvas_placeholder') and self._canvas_placeholder is not None:
                # First time: replace placeholder
                layout.replaceWidget(self._canvas_placeholder, self._mpl_toolbar)
                self._canvas_placeholder.deleteLater()
                self._canvas_placeholder = None
                layout.insertWidget(2, self.canvas, stretch=1)
            else:
                # Canvas already exists, insert toolbar and canvas
                layout.insertWidget(1, self._mpl_toolbar)
                layout.insertWidget(2, self.canvas, stretch=1)

            # Draw the canvas
            self.canvas.draw()
            logger.info("Matplotlib figure displayed successfully")

        except Exception as e:
            logger.error(f"Error in show_matplotlib_figure: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_current_figure(self):
        """
        Get the currently displayed figure.

        Returns
        -------
        matplotlib.figure.Figure or None
            Current figure, or None if no figure is displayed
        """
        return self.current_fig

    def clear_plot(self):
        """Clear the current plot and show welcome message."""
        self.current_fig = None
        self.current_data = None
        self._show_welcome_message()
