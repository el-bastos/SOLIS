"""
Variable Study Widget

Provides interface for studying the relationship between amplitude (α/A parameter)
and absorbance or excitation intensity across multiple datasets.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from typing import List, Dict, Any
import numpy as np
from utils.logger_config import get_logger
from gui.stylesheets import TABLE_STYLE, INFO_LABEL_STYLE, INFO_LABEL_STYLE_SMALL

logger = get_logger(__name__)


class VariableStudyWidget(QWidget):
    """Widget for Variable Study analysis (α vs absorption/intensity)."""

    # Signals for requesting plots
    plot_combined_emission_requested = pyqtSignal(list)  # List of selected items
    plot_alpha_vs_absorption_requested = pyqtSignal(list, str)  # List of items, EI unit
    plot_alpha_vs_intensity_requested = pyqtSignal(list, str)  # List of items, EI unit

    def __init__(self, data_browser=None):
        super().__init__()

        self.data_browser = data_browser
        self.main_window = None  # Will be set by main GUI
        self.selected_items = []

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # === Info Label ===
        info_label = QLabel(
            "Select files in the Data Browser (checkbox column) to include in Variable Study analysis.\n"
            "After analysis is complete, use the buttons below to create plots."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(INFO_LABEL_STYLE)
        layout.addWidget(info_label)

        # === Selected Files Group ===
        files_group = QGroupBox("Selected Files")
        files_layout = QVBoxLayout()

        # Refresh button
        refresh_btn_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Selection")
        self.refresh_button.setToolTip("Update the list of selected files from Data Browser")
        refresh_btn_layout.addWidget(self.refresh_button)
        refresh_btn_layout.addStretch()
        files_layout.addLayout(refresh_btn_layout)

        # Table showing selected files
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(5)
        self.files_table.setHorizontalHeaderLabels(["Compound", "File", "A(λ)", "EI", "α (Mean)"])
        self.files_table.setAlternatingRowColors(False)  # CRITICAL: Disable to prevent platform hover effects
        self.files_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.files_table.setStyleSheet(TABLE_STYLE)  # Apply centralized styling with no hover effects
        # Set column widths
        self.files_table.setColumnWidth(0, 120)  # Compound
        self.files_table.setColumnWidth(1, 180)  # File
        self.files_table.setColumnWidth(2, 70)   # A(λ)
        self.files_table.setColumnWidth(3, 80)   # EI
        self.files_table.setColumnWidth(4, 100)  # α
        files_layout.addWidget(self.files_table)

        files_group.setLayout(files_layout)
        layout.addWidget(files_group)

        # === EI Unit Group ===
        ei_group = QGroupBox("Excitation Intensity Unit")
        ei_layout = QHBoxLayout()

        ei_layout.addWidget(QLabel("Unit:"))
        self.ei_unit_input = QLineEdit()
        self.ei_unit_input.setPlaceholderText("e.g., mW/cm², μJ, nJ/pulse")
        self.ei_unit_input.setToolTip("Specify the unit for excitation intensity (for axis labels)")
        self.ei_unit_input.setMinimumWidth(150)
        ei_layout.addWidget(self.ei_unit_input)
        ei_layout.addStretch()

        ei_group.setLayout(ei_layout)
        layout.addWidget(ei_group)

        # === Plot Buttons Group ===
        plot_group = QGroupBox("Generate Plots")
        plot_layout = QVBoxLayout()

        # Combined emission plot button
        self.plot_emission_button = QPushButton("Plot Combined Emission (All Selected)")
        self.plot_emission_button.setToolTip(
            "Create overlay plot showing mean emission curves and fits for all selected files,\n"
            "with separate residual panels for each dataset"
        )
        self.plot_emission_button.setEnabled(False)
        plot_layout.addWidget(self.plot_emission_button)

        # α vs absorption plot button
        self.plot_alpha_abs_button = QPushButton("Plot α vs (1 - 10^(-A(λ)))")
        self.plot_alpha_abs_button.setToolTip(
            "Plot amplitude (α) vs absorbed light fraction (Beer-Lambert law)\n"
            "with linear regression forced through origin"
        )
        self.plot_alpha_abs_button.setEnabled(False)
        plot_layout.addWidget(self.plot_alpha_abs_button)

        # α vs intensity plot button
        self.plot_alpha_ei_button = QPushButton("Plot α vs Excitation Intensity")
        self.plot_alpha_ei_button.setToolTip(
            "Plot amplitude (α) vs excitation intensity\n"
            "with linear regression forced through origin"
        )
        self.plot_alpha_ei_button.setEnabled(False)
        plot_layout.addWidget(self.plot_alpha_ei_button)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # === Statistics Display (placeholder for future) ===
        stats_group = QGroupBox("Linear Regression Results")
        stats_layout = QVBoxLayout()

        self.stats_label = QLabel("No analysis performed yet.")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet(INFO_LABEL_STYLE_SMALL)
        stats_layout.addWidget(self.stats_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Add stretch at bottom
        layout.addStretch()

    def _connect_signals(self):
        """Connect widget signals."""
        self.refresh_button.clicked.connect(self.refresh_selection)
        self.plot_emission_button.clicked.connect(self._on_plot_emission_clicked)
        self.plot_alpha_abs_button.clicked.connect(self._on_plot_alpha_abs_clicked)
        self.plot_alpha_ei_button.clicked.connect(self._on_plot_alpha_ei_clicked)

    def update_regression_stats(self, stats: Dict[str, Any], plot_type: str):
        """
        Update the statistics display with regression results.

        Parameters
        ----------
        stats : dict
            Regression statistics from plotter
        plot_type : str
            Type of plot ('absorption' or 'intensity')
        """
        if not stats or not stats.get('success'):
            self.stats_label.setText("Regression failed - check data quality.")
            return

        # Extract key statistics
        slope = stats.get('slope', 0)
        intercept = stats.get('intercept', 0)
        r_squared = stats.get('r_squared', 0)
        n_total = stats.get('n_total', 0)
        n_linear = stats.get('n_linear', 0)
        model_type = stats.get('model_type', 'unknown')
        warning = stats.get('warning')

        # Build display text
        plot_name = "α vs (1 - 10^(-A(λ)))" if plot_type == 'absorption' else "α vs Excitation Intensity"

        text = f"<b>{plot_name}</b><br>"
        text += f"<br>"
        text += f"<b>Linear Regression (Iterative Outlier Removal):</b><br>"
        text += f"  Slope: {slope:.2f}<br>"
        text += f"  Intercept: 0.0 (forced through origin)<br>"
        text += f"  R²: {r_squared:.4f}<br>"
        text += f"  Linear range: {n_linear}/{n_total} points<br>"

        # Show outlier removal details
        n_outliers = stats.get('n_outliers', 0)
        n_iterations = stats.get('n_iterations', 0)
        if n_outliers > 0:
            text += f"  Outliers removed: {n_outliers} point(s) in {n_iterations} iteration(s)<br>"
        else:
            text += f"  No outliers detected<br>"

        if warning:
            text += f"<br><b style='color: #CC6600;'>⚠ {warning}</b>"

        self.stats_label.setText(text)
        self.stats_label.setTextFormat(Qt.TextFormat.RichText)

    def set_data_browser(self, data_browser):
        """Set reference to data browser widget."""
        self.data_browser = data_browser

    def set_main_window(self, main_window):
        """Set reference to main window."""
        self.main_window = main_window
        self.data_browser = main_window.data_browser if main_window else None

    def _build_selected_items_from_main_window(self):
        """Build selected_items list from main window's loaded compounds and results."""
        selected_items = []

        if not self.main_window or not self.main_window.loaded_compounds:
            return selected_items

        # Get analysis results from results widget
        kinetics_results = {}
        statistics_results = {}
        if self.main_window.results_widget:
            kinetics_results = self.main_window.results_widget.kinetics_results
            statistics_results = self.main_window.results_widget.statistics_results

        # Build items from all compounds (use all analyzed data for linearity check)
        # loaded_compounds is Dict[compound_name, List[ParsedFile]]
        for compound_name, parsed_files in self.main_window.loaded_compounds.items():
            # Separate decay and abs files
            decay_files = [f for f in parsed_files if f.file_type == 'decay']
            abs_files = [f for f in parsed_files if f.file_type == 'absorption']

            if not decay_files:
                continue

            first_decay = decay_files[0]

            # Get abs value - use the absorbance already linked to decay file
            abs_str = '—'
            abs_value = first_decay.absorbance_at_wavelength
            if abs_value is not None:
                if isinstance(abs_value, list) and len(abs_value) > 0:
                    abs_str = f"{abs_value[0]:.3f}"
                elif isinstance(abs_value, (int, float)):
                    abs_str = f"{abs_value:.3f}"

            # Get EI value
            ei_str = '—'
            if hasattr(first_decay, 'excitation_intensity') and first_decay.excitation_intensity is not None:
                ei_value = first_decay.excitation_intensity
                # Correct attribute name is 'intensity_unit', not 'excitation_intensity_unit'
                ei_unit = first_decay.intensity_unit if hasattr(first_decay, 'intensity_unit') and first_decay.intensity_unit else 'a.u.'
                ei_str = f"{ei_value} {ei_unit}"

            # Get analysis result (use statistics if available, otherwise kinetics)
            analysis_result = None
            use_statistics = False
            if compound_name in statistics_results:
                analysis_result = statistics_results[compound_name]
                use_statistics = True
            elif compound_name in kinetics_results:
                analysis_result = kinetics_results[compound_name]

            # Build item dict with all required fields for compatibility
            # This needs to work with both the table display AND the plot functions
            item = {
                # Basic info
                'compound': compound_name,
                'file_name': first_decay.file_path.name if hasattr(first_decay.file_path, 'name') else str(first_decay.file_path),
                'parsed_file': first_decay,  # For table display

                # Metadata
                'abs_at_wavelength': abs_str,
                'excitation_intensity': ei_str,
                'wavelength': first_decay.wavelength,
                'classification': first_decay.classification,

                # Analysis results
                'analysis_result': analysis_result,

                # For merged plot compatibility (same format as results_widget uses)
                'type': 'mean' if use_statistics else 'replicate',
            }

            # Add fields specific to type
            if use_statistics and analysis_result:
                # Add mean_arrays and sd_arrays for merged plot
                item['mean_arrays'] = analysis_result.get('mean_arrays', {})
                item['sd_arrays'] = analysis_result.get('sd_arrays', {})
            elif analysis_result and isinstance(analysis_result, dict):
                # This is kinetics_results which has 'results' list
                item['results'] = analysis_result.get('results', [])
                if item['results']:
                    item['result'] = item['results'][0]  # Use first result for single replicate
                    item['replicate_num'] = 1

            selected_items.append(item)

        return selected_items

    def refresh_selection(self):
        """Refresh the list of selected files from Data Browser."""
        if not self.main_window:
            QMessageBox.warning(self, "Error", "Main window not available.")
            return

        # Build selected_items from main window's data
        self.selected_items = self._build_selected_items_from_main_window()

        # Update table
        self._update_files_table()

        # Enable/disable buttons based on selection AND which variable is changing
        has_selection = len(self.selected_items) > 0
        has_analysis = any(item.get('analysis_result') is not None for item in self.selected_items)

        self.plot_emission_button.setEnabled(has_selection and has_analysis)

        # Check which variables are actually varying
        if len(self.selected_items) >= 2:
            # Extract A(λ) values
            abs_values = []
            for item in self.selected_items:
                abs_str = item.get('abs_at_wavelength', '—')
                if abs_str != '—':
                    try:
                        abs_values.append(float(abs_str))
                    except ValueError:
                        pass

            # Extract EI values
            ei_values = []
            for item in self.selected_items:
                ei_str = item.get('excitation_intensity', '—')
                if ei_str != '—':
                    try:
                        ei_parts = ei_str.strip().split()
                        if len(ei_parts) >= 1:
                            ei_values.append(float(ei_parts[0]))
                    except (ValueError, IndexError):
                        pass

            # Calculate coefficient of variation (CV) to detect which variable is changing
            # CV = std / mean - if CV < 0.1 (10%), consider it constant
            abs_is_varying = False
            ei_is_varying = False
            abs_cv = 0.0
            ei_cv = 0.0

            if len(abs_values) >= 2:
                abs_mean = np.mean(abs_values)
                abs_std = np.std(abs_values)
                abs_cv = abs_std / abs_mean if abs_mean > 0 else 0
                abs_is_varying = abs_cv > 0.10  # More than 10% variation

            if len(ei_values) >= 2:
                ei_mean = np.mean(ei_values)
                ei_std = np.std(ei_values)
                ei_cv = ei_std / ei_mean if ei_mean > 0 else 0
                ei_is_varying = ei_cv > 0.10  # More than 10% variation

            # Enable buttons only if the variable is actually varying
            # Convert numpy bool to Python bool for Qt compatibility
            self.plot_alpha_abs_button.setEnabled(bool(abs_is_varying and has_analysis and len(abs_values) >= 2))
            self.plot_alpha_ei_button.setEnabled(bool(ei_is_varying and has_analysis and len(ei_values) >= 2))

            # Update tooltips to explain why disabled
            if not abs_is_varying and len(abs_values) >= 2:
                self.plot_alpha_abs_button.setToolTip(
                    f"A(λ) is nearly constant (CV={abs_cv:.1%}). Cannot perform correlation analysis."
                )
            elif len(abs_values) < 2:
                self.plot_alpha_abs_button.setToolTip("Need at least 2 files with A(λ) data")
            else:
                self.plot_alpha_abs_button.setToolTip(
                    "Plot amplitude (α) vs absorbed light fraction (Beer-Lambert law)\n"
                    "with linear regression"
                )

            if not ei_is_varying and len(ei_values) >= 2:
                self.plot_alpha_ei_button.setToolTip(
                    f"EI is nearly constant (CV={ei_cv:.1%}). Cannot perform correlation analysis."
                )
            elif len(ei_values) < 2:
                self.plot_alpha_ei_button.setToolTip("Need at least 2 files with EI data")
            else:
                self.plot_alpha_ei_button.setToolTip(
                    "Plot amplitude (α) vs excitation intensity\n"
                    "with linear regression"
                )
        else:
            # Not enough data points
            self.plot_alpha_abs_button.setEnabled(False)
            self.plot_alpha_ei_button.setEnabled(False)
            self.plot_alpha_abs_button.setToolTip("Need at least 2 files for correlation analysis")
            self.plot_alpha_ei_button.setToolTip("Need at least 2 files for correlation analysis")

    def _update_files_table(self):
        """Update the table showing selected files."""
        self.files_table.setRowCount(0)

        for item in self.selected_items:
            row = self.files_table.rowCount()
            self.files_table.insertRow(row)

            # Compound
            compound = item.get('compound', 'Unknown')
            self.files_table.setItem(row, 0, QTableWidgetItem(compound))

            # File name
            parsed_file = item.get('parsed_file')
            if parsed_file:
                from pathlib import Path
                file_name = Path(parsed_file.file_path).stem
                self.files_table.setItem(row, 1, QTableWidgetItem(file_name))
            else:
                self.files_table.setItem(row, 1, QTableWidgetItem("N/A"))

            # A(λ)
            abs_value = item.get('abs_at_wavelength', '—')
            self.files_table.setItem(row, 2, QTableWidgetItem(str(abs_value)))

            # EI
            ei_value = item.get('excitation_intensity', '—')
            self.files_table.setItem(row, 3, QTableWidgetItem(str(ei_value)))

            # α (Mean A parameter from analysis)
            analysis_result = item.get('analysis_result')
            if analysis_result:
                try:
                    # Get mean A from analysis results
                    # analysis_result could be a list of KineticsResult or summary dict
                    if isinstance(analysis_result, list):
                        # List of KineticsResult - calculate mean
                        A_values = [r.parameters.A for r in analysis_result if hasattr(r, 'parameters')]
                        if A_values:
                            A_mean = np.mean(A_values)
                            A_sd = np.std(A_values, ddof=1) if len(A_values) > 1 else 0
                            alpha_str = f"{A_mean:.1f} ± {A_sd:.1f}"
                        else:
                            alpha_str = "N/A"
                    elif isinstance(analysis_result, dict):
                        # Summary dict - could be statistics_results or kinetics_results
                        # statistics_results: {'statistics': {...}, 'n_replicates': ...}
                        # kinetics_results: {'results': [...], 'wavelength': ...}

                        if 'statistics' in analysis_result:
                            # This is from statistics_results
                            stats = analysis_result['statistics']
                            A_mean = stats.get('A_mean', None)
                            A_sd = stats.get('A_sd', None)
                            if A_mean is not None and A_mean != 'ND':
                                if A_sd is not None and A_sd != 'ND':
                                    alpha_str = f"{A_mean:.1f} ± {A_sd:.1f}"
                                else:
                                    alpha_str = f"{A_mean:.1f}"
                            else:
                                alpha_str = "N/A"
                        elif 'results' in analysis_result:
                            # This is from kinetics_results
                            results_list = analysis_result['results']
                            A_values = [r.parameters.A for r in results_list if hasattr(r, 'parameters')]
                            if A_values:
                                A_mean = np.mean(A_values)
                                A_sd = np.std(A_values, ddof=1) if len(A_values) > 1 else 0
                                alpha_str = f"{A_mean:.1f} ± {A_sd:.1f}"
                            else:
                                alpha_str = "N/A"
                        else:
                            alpha_str = "N/A"
                    else:
                        # Single KineticsResult
                        if hasattr(analysis_result, 'parameters'):
                            alpha_str = f"{analysis_result.parameters.A:.1f}"
                        else:
                            alpha_str = "N/A"
                except Exception as e:
                    logger.error(f"Error extracting α: {e}")
                    alpha_str = "Error"
            else:
                alpha_str = "Not analyzed"

            self.files_table.setItem(row, 4, QTableWidgetItem(alpha_str))

        # Update info label
        n_selected = len(self.selected_items)
        n_analyzed = sum(1 for item in self.selected_items if item.get('analysis_result') is not None)

        if n_selected == 0:
            info_text = "No files selected. Check the 'Select' column in Data Browser."
        else:
            info_text = f"{n_selected} file(s) selected, {n_analyzed} analyzed."

        # Find info label and update it
        # (We'll just create a simple summary for now)

    def _on_plot_emission_clicked(self):
        """Handle plot combined emission button click."""
        if len(self.selected_items) == 0:
            QMessageBox.warning(self, "No Selection", "Please select files in the Data Browser.")
            return

        # Emit signal with selected items
        self.plot_combined_emission_requested.emit(self.selected_items)

    def _on_plot_alpha_abs_clicked(self):
        """Handle plot α vs absorption button click."""
        if len(self.selected_items) < 2:
            QMessageBox.warning(
                self,
                "Insufficient Data",
                "Need at least 2 files with A(λ) data for correlation analysis."
            )
            return

        # Get EI unit (not used for this plot, but keep consistent API)
        ei_unit = self.ei_unit_input.text().strip() or "a.u."

        # Emit signal
        self.plot_alpha_vs_absorption_requested.emit(self.selected_items, ei_unit)

    def _on_plot_alpha_ei_clicked(self):
        """Handle plot α vs intensity button click."""
        if len(self.selected_items) < 2:
            QMessageBox.warning(
                self,
                "Insufficient Data",
                "Need at least 2 files with EI data for correlation analysis."
            )
            return

        # Get EI unit
        ei_unit = self.ei_unit_input.text().strip()
        if not ei_unit:
            QMessageBox.warning(
                self,
                "Missing Unit",
                "Please specify the excitation intensity unit."
            )
            return

        # Emit signal
        self.plot_alpha_vs_intensity_requested.emit(self.selected_items, ei_unit)

    def update_statistics(self, stats_dict: Dict[str, Any]):
        """
        Update the statistics display with regression results.

        Parameters
        ----------
        stats_dict : dict
            Dictionary containing regression statistics (slope, R², etc.)
        """
        stats_text = "Linear Regression Results:\n\n"

        # Model type
        model_type = stats_dict.get('model_type', 'unknown')
        if model_type == 'forced_zero':
            stats_text += "Model: Forced through origin\n"
        elif model_type == 'free_intercept':
            stats_text += "Model: Free intercept\n"

        # Intercept (if not zero)
        intercept = stats_dict.get('intercept', 0)
        if abs(intercept) > 1e-10:
            stats_text += f"Intercept: {intercept:.2f}\n"

        # Slope
        if 'slope' in stats_dict:
            stats_text += f"Slope: {stats_dict['slope']:.3e}\n"

        # R²
        if 'r_squared' in stats_dict:
            stats_text += f"R²: {stats_dict['r_squared']:.4f}\n"

        # Show both R² values if available
        if 'r_squared_zero' in stats_dict and 'r_squared_free' in stats_dict:
            stats_text += f"  (R²_zero={stats_dict['r_squared_zero']:.3f}, "
            stats_text += f"R²_free={stats_dict['r_squared_free']:.3f})\n"

        # Points
        if 'n_total' in stats_dict:
            stats_text += f"Points: {stats_dict['n_total']}\n"

        if 'n_linear' in stats_dict and 'n_total' in stats_dict:
            stats_text += f"Linear range: {stats_dict['n_linear']}/{stats_dict['n_total']} points\n"

        # Warning (if present)
        if 'warning' in stats_dict and stats_dict['warning']:
            stats_text += f"\n⚠ WARNING:\n{stats_dict['warning']}\n"

        self.stats_label.setText(stats_text)
