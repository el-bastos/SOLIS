#!/usr/bin/env python3
"""
Results Viewer Widget - Right panel with result tabs

Contains tabs for:
- Kinetics results
- Quantum Yield results
- Surplus results
- Heterogeneous results
- Plot tabs (matplotlib embedded)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QHeaderView, QAbstractItemView, QTableWidget, QTableWidgetItem,
    QStyle
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon
from typing import Dict, List
import numpy as np
from pathlib import Path

from utils.logger_config import get_logger
from gui.stylesheets import TREE_STYLE, TABLE_STYLE, get_tab_close_button_style

logger = get_logger(__name__)


class ResultsViewerWidget(QWidget):
    """
    Results viewer with tabs - RIGHT PANEL

    Contains tabs for analysis results and plots.
    """

    # Signals
    status_message = pyqtSignal(str)
    plot_requested = pyqtSignal(list)  # Individual plots
    plot_merged_requested = pyqtSignal(list)  # Merged plots
    surplus_plot_requested = pyqtSignal(str)  # Surplus plot
    heterogeneous_plot_requested = pyqtSignal(str)  # Heterogeneous plots

    # Kinetics tree column indices
    COL_COMPOUND = 0
    COL_WAVELENGTH = 1
    COL_CLASSIFICATION = 2
    COL_A = 3
    COL_TAU_DELTA = 4
    COL_TAU_T = 5
    COL_SNR = 6
    COL_R_SQUARED = 7
    COL_CHI_SQUARED = 8
    COL_MASKED_TIME = 9
    COL_T0 = 10
    COL_Y0 = 11

    def __init__(self, parent=None):
        super().__init__(parent)

        # Storage for results
        self.kinetics_results = {}
        self.statistics_results = {}
        self.qy_results = []
        self.surplus_results = {}
        self.heterogeneous_results = {}

        # Track which tabs are open
        self.tab_indices = {}  # {'Kinetics': tab_index, 'plot_name': tab_index, ...}

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI with tab widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # RESULTS label (matching BROWSER label style)
        results_label = QLabel("RESULTS")
        results_label.setStyleSheet("color: rgb(120, 120, 120); font-weight: bold; padding-left: 5px;")
        font = results_label.font()
        font.setPointSize(font.pointSize() - 1)  # Slightly smaller
        results_label.setFont(font)
        layout.addWidget(results_label)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)

        # Set uniform icon size for tab buttons (close, etc.)
        self.tabs.setIconSize(QSize(16, 16))

        # Load custom close icon
        icon_dir = Path(__file__).parent.parent / 'icons'
        close_icon_path = icon_dir / 'close_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg'
        if close_icon_path.exists():
            # Apply custom close button icon via centralized stylesheet
            self.tabs.setStyleSheet(get_tab_close_button_style(close_icon_path))

        layout.addWidget(self.tabs)

    def _close_tab(self, index: int):
        """Handle tab close."""
        # Get tab name
        tab_name = self.tabs.tabText(index)

        # Remove from tracking dict
        if tab_name in self.tab_indices:
            del self.tab_indices[tab_name]

        # Remove tab
        self.tabs.removeTab(index)

        # Update ALL indices for tabs that were after the closed one
        for key in list(self.tab_indices.keys()):
            if self.tab_indices[key] > index:
                self.tab_indices[key] -= 1

    # ==================== TAB CREATION ====================

    def open_result_tab(self, result_type: str):
        """Open or switch to result tab."""
        # Check if tab already open
        if result_type in self.tab_indices:
            tab_index = self.tab_indices[result_type]
            # Verify tab still exists
            if tab_index < self.tabs.count():
                self.tabs.setCurrentIndex(tab_index)
                return

        # Create new result tab
        if result_type == 'Kinetics':
            tab_widget = self._create_kinetics_tab()
        elif result_type == 'Quantum Yield':
            tab_widget = self._create_qy_tab()
        elif result_type == 'Surplus':
            tab_widget = self._create_surplus_tab()
        elif result_type == 'Heterogeneous':
            tab_widget = self._create_heterogeneous_tab()
        else:
            logger.warning(f"Unknown result type: {result_type}")
            return

        # Add tab
        tab_index = self.tabs.addTab(tab_widget, result_type)
        self.tabs.setCurrentIndex(tab_index)
        self.tab_indices[result_type] = tab_index

    def open_plot_tab(self, plot_name: str, plot_data: dict):
        """Open or switch to plot tab."""
        # Check if tab already open
        if plot_name in self.tab_indices:
            tab_index = self.tab_indices[plot_name]
            # Verify tab still exists
            if tab_index < self.tabs.count():
                self.tabs.setCurrentIndex(tab_index)
                return

        # Create new plot tab
        tab_widget = self._create_plot_tab_widget(plot_data)

        # Add tab
        tab_index = self.tabs.addTab(tab_widget, plot_name)
        self.tabs.setCurrentIndex(tab_index)
        self.tab_indices[plot_name] = tab_index

    def _create_kinetics_tab(self) -> QWidget:
        """Create Kinetics results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create tree with ExtendedSelection
        tree = QTreeWidget()
        tree.setColumnCount(12)
        tree.setHeaderLabels([
            "Compound / Replicate", "λ (nm)", "Classification",
            "A", "τΔ (μs)", "τT (μs)", "SNR", "R²", "χ²ᵣ", "Masked (μs)", "t0 (μs)", "y0"
        ])
        tree.setAlternatingRowColors(False)  # CRITICAL: Disable to prevent platform hover effects
        tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        tree.setStyleSheet(TREE_STYLE)

        # Auto-resize columns
        header = tree.header()
        for col in range(12):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        # Store reference
        self.kinetics_tree = tree

        layout.addWidget(tree)

        # Populate if data exists
        if self.kinetics_results and self.statistics_results:
            self._populate_kinetics_tree()

        return widget

    def _create_qy_tab(self) -> QWidget:
        """Create Quantum Yield results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create table widget
        self.qy_table = QTableWidget()
        self.qy_table.setColumnCount(6)
        self.qy_table.setHorizontalHeaderLabels([
            "Sample", "Standard", "λ (nm)", "Φ", "Error", "Rel. Error (%)"
        ])
        self.qy_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.qy_table.setAlternatingRowColors(False)  # CRITICAL: Disable to prevent platform hover effects
        self.qy_table.verticalHeader().setVisible(False)

        # LEFT-align headers
        for col in range(6):
            item = self.qy_table.horizontalHeaderItem(col)
            if item:
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Set selection color, disable hover
        self.qy_table.setStyleSheet(TABLE_STYLE)

        # Auto-resize columns
        header = self.qy_table.horizontalHeader()
        for col in range(6):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.qy_table)

        # Populate if data exists
        if self.qy_results:
            self._populate_qy_table()

        return widget

    def _create_surplus_tab(self) -> QWidget:
        """Create Surplus results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create table widget
        self.surplus_table = QTableWidget()
        self.surplus_table.setColumnCount(8)
        self.surplus_table.setHorizontalHeaderLabels([
            "Compound", "α", "β", "τΔ,1 (μs)", "τΔ,2 (μs)", "τT (μs)", "R²", "Mask (μs)"
        ])
        self.surplus_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.surplus_table.setAlternatingRowColors(False)  # CRITICAL: Disable to prevent platform hover effects
        self.surplus_table.verticalHeader().setVisible(False)

        # LEFT-align headers
        for col in range(8):
            item = self.surplus_table.horizontalHeaderItem(col)
            if item:
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Set selection color, disable hover
        self.surplus_table.setStyleSheet(TABLE_STYLE)

        # Auto-resize columns
        header = self.surplus_table.horizontalHeader()
        for col in range(8):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.surplus_table)

        # Populate if data exists
        if self.surplus_results:
            self._populate_surplus_table()

        return widget

    def _create_heterogeneous_tab(self) -> QWidget:
        """Create Heterogeneous results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create table widget
        self.heterogeneous_table = QTableWidget()
        self.heterogeneous_table.setColumnCount(11)
        self.heterogeneous_table.setHorizontalHeaderLabels([
            "Compound", "Rep", "τT (μs)", "τw (μs)", "τL (μs)", "A", "B", "C", "A/B", "χ²red", "Action"
        ])
        self.heterogeneous_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.heterogeneous_table.setAlternatingRowColors(False)  # CRITICAL: Disable to prevent platform hover effects
        self.heterogeneous_table.verticalHeader().setVisible(False)

        # LEFT-align headers
        for col in range(11):
            item = self.heterogeneous_table.horizontalHeaderItem(col)
            if item:
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Set selection color, disable hover
        self.heterogeneous_table.setStyleSheet(TABLE_STYLE)

        # Auto-resize columns
        header = self.heterogeneous_table.horizontalHeader()
        for col in range(11):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.heterogeneous_table)

        # Populate if data exists
        if self.heterogeneous_results:
            self._populate_heterogeneous_table()

        return widget

    def _create_plot_tab_widget(self, plot_data: dict) -> QWidget:
        """Create widget to display matplotlib plot with control buttons."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Check if we have a matplotlib figure
        fig = plot_data.get('figure')
        if fig is not None:
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
                from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

                # Create canvas
                canvas = FigureCanvasQTAgg(fig)

                # Create matplotlib toolbar
                mpl_toolbar = NavigationToolbar2QT(canvas, widget)

                # Create custom control toolbar
                control_toolbar = QWidget()
                control_layout = QHBoxLayout(control_toolbar)
                control_layout.setContentsMargins(5, 5, 5, 5)

                # Save CSV button
                btn_save_csv = QPushButton("Save CSV")
                btn_save_csv.clicked.connect(lambda: self._export_plot_csv(fig, plot_data))
                control_layout.addWidget(btn_save_csv)

                # Save PDF button
                btn_save_pdf = QPushButton("Save PDF")
                btn_save_pdf.clicked.connect(lambda: self._export_plot_pdf(fig, plot_data))
                control_layout.addWidget(btn_save_pdf)

                # Log X toggle button (only for decay plots, not absorption)
                plot_name = plot_data.get('name', '')
                is_absorption_plot = plot_name.startswith('abs_')
                if not is_absorption_plot:
                    btn_log_x = QPushButton("Log X")
                    btn_log_x.setCheckable(True)

                    # Check current X-axis scale
                    is_log_scale = False
                    if fig.get_axes():
                        first_ax = fig.get_axes()[0]
                        is_log_scale = first_ax.get_xscale() == 'log'
                    btn_log_x.setChecked(is_log_scale)

                    btn_log_x.clicked.connect(lambda checked: self._toggle_log_x(fig, canvas, checked))
                    control_layout.addWidget(btn_log_x)

                control_layout.addStretch()

                # Add to layout
                layout.addWidget(control_toolbar)
                layout.addWidget(mpl_toolbar)
                layout.addWidget(canvas)

                # Store references
                widget._canvas = canvas
                widget._figure = fig
                widget._plot_data = plot_data

            except Exception as e:
                logger.error(f"Failed to embed matplotlib figure: {e}")
                label = QLabel(f"Error embedding plot: {e}")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
        else:
            # Fallback
            label = QLabel(f"Plot: {plot_data.get('name', 'Unknown')}\n(No figure data)")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

        return widget

    # ==================== RESULTS POPULATION ====================

    def populate_kinetics_results(self, kinetics_results: Dict, statistics_results: Dict):
        """Populate kinetics results."""
        self.kinetics_results = kinetics_results
        self.statistics_results = statistics_results

        # If kinetics tab is already open, populate it
        if hasattr(self, 'kinetics_tree') and self.kinetics_tree is not None:
            self._populate_kinetics_tree()

    def _populate_kinetics_tree(self):
        """Populate the kinetics tree widget."""
        self.kinetics_tree.setUpdatesEnabled(False)

        try:
            self.kinetics_tree.clear()

            for compound_name in sorted(self.kinetics_results.keys()):
                compound_data = self.kinetics_results[compound_name]
                results_list = compound_data['results']
                wavelength = compound_data.get('wavelength', '—')
                classification = compound_data.get('classification', '—')
                stats_data = self.statistics_results.get(compound_name)

                # Create parent item
                parent_item = self._create_compound_tree_item(
                    compound_name, wavelength, classification, stats_data
                )
                self.kinetics_tree.addTopLevelItem(parent_item)

                # Add replicates as children
                for rep_num, result in enumerate(results_list, start=1):
                    replicate_item = self._create_kinetics_replicate_item(
                        compound_name, rep_num, result
                    )
                    parent_item.addChild(replicate_item)

                parent_item.setExpanded(False)

        finally:
            self.kinetics_tree.setUpdatesEnabled(True)

    def _create_compound_tree_item(self, compound_name: str, wavelength,
                                   classification, stats_data: Dict) -> QTreeWidgetItem:
        """Create parent tree item for compound (shows mean ± SD)."""
        item = QTreeWidgetItem()

        # Compound name with replicate count
        stats = stats_data['statistics'] if stats_data else {}
        n_reps = stats_data.get('n_replicates', 0) if stats_data else 0
        item.setText(self.COL_COMPOUND, f"{compound_name} (n={n_reps})")

        # Wavelength
        if isinstance(wavelength, (int, float)):
            item.setText(self.COL_WAVELENGTH, f"{wavelength:.0f}")
        else:
            item.setText(self.COL_WAVELENGTH, str(wavelength))

        # Classification
        item.setText(self.COL_CLASSIFICATION, str(classification))

        if stats_data and stats:
            # Populate statistics columns
            A_mean = stats.get('A_mean', 0)
            A_sd = stats.get('A_sd', 0)
            item.setText(self.COL_A, f"{A_mean:.1f} ± {A_sd:.1f}")

            tau_d_mean = stats.get('tau_delta_mean', 0)
            tau_d_sd = stats.get('tau_delta_sd', 0)
            item.setText(self.COL_TAU_DELTA, f"{tau_d_mean:.2f} ± {tau_d_sd:.2f}")

            tau_T_mean = stats.get('tau_T_mean', 'ND')
            if isinstance(tau_T_mean, (int, float)):
                tau_T_sd = stats.get('tau_T_sd', 0)
                item.setText(self.COL_TAU_T, f"{tau_T_mean:.2f} ± {tau_T_sd:.2f}")
            else:
                item.setText(self.COL_TAU_T, "ND")

            snr_mean = stats.get('snr_linear_mean', 0)
            snr_sd = stats.get('snr_linear_sd', 0)
            item.setText(self.COL_SNR, f"{snr_mean:.1f}:1 ± {snr_sd:.1f}")

            r2_mean = stats.get('r_squared_mean', 0)
            r2_sd = stats.get('r_squared_sd', 0)
            item.setText(self.COL_R_SQUARED, f"{r2_mean:.4f} ± {r2_sd:.4f}")

            chi2_mean = stats.get('reduced_chi_square_mean', 0)
            chi2_sd = stats.get('reduced_chi_square_sd', 0)
            item.setText(self.COL_CHI_SQUARED, f"{chi2_mean:.3f} ± {chi2_sd:.3f}")

            # Masked time
            mean_arrays = stats_data.get('mean_arrays', {})
            if 'fitting_mask' in mean_arrays and 'time_experiment_us' in mean_arrays:
                mask = mean_arrays['fitting_mask']
                time = mean_arrays['time_experiment_us']
                masked_indices = np.where(mask)[0]
                if len(masked_indices) > 0:
                    masked_time = time[masked_indices[0]]
                    item.setText(self.COL_MASKED_TIME, f"{masked_time:.3f}")
                else:
                    item.setText(self.COL_MASKED_TIME, "—")
            else:
                item.setText(self.COL_MASKED_TIME, "—")

            t0_mean = stats.get('t0_mean', 0)
            t0_sd = stats.get('t0_sd', 0)
            item.setText(self.COL_T0, f"{t0_mean:.4f} ± {t0_sd:.4f}")

            y0_mean = stats.get('y0_mean', 0)
            y0_sd = stats.get('y0_sd', 0)
            item.setText(self.COL_Y0, f"{y0_mean:.3f} ± {y0_sd:.3f}")

        # Bold font
        font = item.font(self.COL_COMPOUND)
        font.setBold(True)
        for col in range(12):
            item.setFont(col, font)

        # Store data
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'compound',
            'compound': compound_name
        })

        return item

    def _create_kinetics_replicate_item(self, compound_name: str, rep_num: int,
                                       result) -> QTreeWidgetItem:
        """Create child tree item for individual replicate."""
        item = QTreeWidgetItem()

        item.setText(self.COL_COMPOUND, f"  Replicate {rep_num}")
        item.setText(self.COL_WAVELENGTH, "")
        item.setText(self.COL_CLASSIFICATION, "")
        item.setText(self.COL_A, f"{result.parameters.A:.1f}")
        item.setText(self.COL_TAU_DELTA, f"{result.parameters.tau_delta:.2f}")

        if isinstance(result.parameters.tau_T, (int, float)):
            item.setText(self.COL_TAU_T, f"{result.parameters.tau_T:.2f}")
        else:
            item.setText(self.COL_TAU_T, "ND")

        snr = result.snr_result.snr_linear if result.snr_result else 0
        item.setText(self.COL_SNR, f"{snr:.1f}:1")
        item.setText(self.COL_R_SQUARED, f"{result.fit_quality.r_squared:.4f}")
        item.setText(self.COL_CHI_SQUARED, f"{result.fit_quality.reduced_chi_square:.3f}")

        masked_indices = np.where(result.fitting_mask)[0]
        if len(masked_indices) > 0:
            masked_time = result.time_experiment_us[masked_indices[0]]
            item.setText(self.COL_MASKED_TIME, f"{masked_time:.3f}")
        else:
            item.setText(self.COL_MASKED_TIME, "—")

        item.setText(self.COL_T0, f"{result.parameters.t0:.4f}")
        item.setText(self.COL_Y0, f"{result.parameters.y0:.3f}")

        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'replicate',
            'compound': compound_name,
            'replicate_num': rep_num,
            'result': result
        })

        return item

    def populate_qy_results(self, qy_results: List):
        """Populate QY results."""
        self.qy_results = qy_results

        # If QY tab is open, populate it
        if hasattr(self, 'qy_table'):
            self._populate_qy_table()

    def _populate_qy_table(self):
        """Populate the QY table widget."""
        self.qy_table.setRowCount(0)

        for qy_data in self.qy_results:
            row = self.qy_table.rowCount()
            self.qy_table.insertRow(row)

            # Sample
            sample = qy_data.get('sample_compound', '—')
            item = QTableWidgetItem(sample)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.qy_table.setItem(row, 0, item)

            # Standard
            standard = qy_data.get('standard_compound', '—')
            item = QTableWidgetItem(standard)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.qy_table.setItem(row, 1, item)

            # Wavelength
            wavelength = qy_data.get('wavelength', '—')
            if isinstance(wavelength, (int, float)):
                item = QTableWidgetItem(f"{wavelength:.0f}")
            else:
                item = QTableWidgetItem(str(wavelength))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.qy_table.setItem(row, 2, item)

            # Quantum Yield (Φ)
            qy_value = qy_data.get('quantum_yield', 0)
            item = QTableWidgetItem(f"{qy_value:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.qy_table.setItem(row, 3, item)

            # Error
            qy_error = qy_data.get('quantum_yield_error', 0)
            item = QTableWidgetItem(f"{qy_error:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.qy_table.setItem(row, 4, item)

            # Relative Error (%)
            if qy_value > 0:
                rel_error = (qy_error / qy_value) * 100
                item = QTableWidgetItem(f"{rel_error:.2f}")
            else:
                item = QTableWidgetItem("—")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.qy_table.setItem(row, 5, item)

    def populate_surplus_results(self, surplus_results: dict):
        """Populate surplus results."""
        self.surplus_results = surplus_results

        # If Surplus tab is open, populate it
        if hasattr(self, 'surplus_table'):
            self._populate_surplus_table()

    def _populate_surplus_table(self):
        """Populate the Surplus table widget."""
        self.surplus_table.setRowCount(0)

        for compound_name, result in self.surplus_results.items():
            row = self.surplus_table.rowCount()
            self.surplus_table.insertRow(row)

            # Compound name
            item = QTableWidgetItem(compound_name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.surplus_table.setItem(row, 0, item)

            # Parameters from final fit
            params = result.final_params
            for col, key in enumerate(['alpha', 'beta', 'tau_delta_1', 'tau_delta_2', 'tau_T'], start=1):
                item = QTableWidgetItem(f"{params[key]:.3f}" if col > 2 else f"{params[key]:.1f}")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.surplus_table.setItem(row, col, item)

            # R²
            item = QTableWidgetItem(f"{result.final_r_squared:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.surplus_table.setItem(row, 6, item)

            # Mask time
            item = QTableWidgetItem(f"{result.mask_time:.1f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.surplus_table.setItem(row, 7, item)

    def populate_heterogeneous_results(self, heterogeneous_results: dict):
        """Populate heterogeneous results."""
        self.heterogeneous_results = heterogeneous_results

        # If Heterogeneous tab is open, populate it
        if hasattr(self, 'heterogeneous_table'):
            self._populate_heterogeneous_table()

    def _populate_heterogeneous_table(self):
        """Populate the Heterogeneous table widget."""
        self.heterogeneous_table.setRowCount(0)

        for key, result in self.heterogeneous_results.items():
            row = self.heterogeneous_table.rowCount()
            self.heterogeneous_table.insertRow(row)

            # Parse key: "{compound_name}_Rep{N}"
            if "_Rep" in key:
                compound_name = key.split("_Rep")[0]
                rep_num = key.split("_Rep")[1]
            else:
                compound_name = key
                rep_num = "1"

            # Compound name
            item = QTableWidgetItem(compound_name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.heterogeneous_table.setItem(row, 0, item)

            # Replicate number
            item = QTableWidgetItem(rep_num)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.heterogeneous_table.setItem(row, 1, item)

            # Fit parameters
            item = QTableWidgetItem(f"{result.tau_T:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.heterogeneous_table.setItem(row, 2, item)

            item = QTableWidgetItem(f"{result.tau_delta_water:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.heterogeneous_table.setItem(row, 3, item)

            item = QTableWidgetItem(f"{result.parameters.tau_delta_lipid:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.heterogeneous_table.setItem(row, 4, item)

            item = QTableWidgetItem(f"{result.amplitude_lipid:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.heterogeneous_table.setItem(row, 5, item)

            item = QTableWidgetItem(f"{result.amplitude_water:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.heterogeneous_table.setItem(row, 6, item)

            item = QTableWidgetItem(f"{result.background:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.heterogeneous_table.setItem(row, 7, item)

            # A/B ratio
            a_over_b = result.rate_ratio
            item = QTableWidgetItem(f"{a_over_b:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.heterogeneous_table.setItem(row, 8, item)

            # χ²red
            item = QTableWidgetItem(f"{result.reduced_chi_square:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.heterogeneous_table.setItem(row, 9, item)

            # Action column: "View Plots" button
            view_button = QPushButton("View Plots")
            view_button.clicked.connect(lambda checked, k=key: self._view_heterogeneous_plots(k))
            self.heterogeneous_table.setCellWidget(row, 10, view_button)

    def _view_heterogeneous_plots(self, key: str):
        """Open heterogeneous analysis plots for a result."""
        if key in self.heterogeneous_results:
            self.status_message.emit(f"Opening heterogeneous plots for {key}...")
            self.heterogeneous_plot_requested.emit(key)

    # ==================== PLOT METHODS ====================

    def get_selected_items_for_plotting(self) -> List:
        """Get selected items from kinetics tree for plotting."""
        selected_items = []

        # Check if kinetics tree exists
        if not hasattr(self, 'kinetics_tree') or self.kinetics_tree is None:
            return selected_items

        # Get selected items from kinetics tree
        tree_selected = self.kinetics_tree.selectedItems()

        for item in tree_selected:
            item_data = item.data(0, Qt.ItemDataRole.UserRole)
            if not item_data:
                continue

            if item_data.get('type') == 'compound':
                # Parent item (mean)
                compound_name = item_data['compound']
                stats_data = self.statistics_results.get(compound_name)
                if stats_data:
                    selected_items.append({
                        'type': 'mean',
                        'compound': compound_name,
                        'mean_arrays': stats_data['mean_arrays'],
                        'sd_arrays': stats_data['sd_arrays']
                    })

            elif item_data.get('type') == 'replicate':
                # Child item (individual replicate)
                selected_items.append({
                    'type': 'replicate',
                    'compound': item_data['compound'],
                    'replicate_num': item_data['replicate_num'],
                    'result': item_data['result']
                })

        return selected_items

    # ==================== EXPORT METHODS ====================

    def _export_plot_csv(self, fig, plot_data: dict):
        """Export plot data to CSV with robust handling of various data structures."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import pandas as pd
        import numpy as np
        import os

        try:
            # Check if figure has export data
            if not hasattr(fig, 'solis_export_data'):
                logger.warning("No export data attached to figure")
                QMessageBox.warning(
                    self,
                    "Export Error",
                    "This plot does not have exportable data attached."
                )
                return

            # Get export data
            export_data_attr = fig.solis_export_data
            if callable(export_data_attr):
                export_data = export_data_attr()
            else:
                export_data = export_data_attr

            # Prompt for save location
            plot_name = plot_data.get('name', 'plot')
            default_filename = f"{plot_name}.csv"

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save CSV",
                default_filename,
                "CSV Files (*.csv)"
            )

            if not file_path:
                return

            # Smart export based on data structure
            df = self._convert_export_data_to_dataframe(export_data)
            df.to_csv(file_path, index=False)

            logger.info(f"Plot data exported to CSV: {file_path}")
            self.status_message.emit(f"CSV saved: {os.path.basename(file_path)}")

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export CSV:\n{e}"
            )

    def _convert_export_data_to_dataframe(self, export_data: dict):
        """
        Convert export data dictionary to a pandas DataFrame.

        Handles various data structures:
        - 2D arrays (chi-square surfaces) -> flattened with grid indices
        - Arrays of different lengths -> padded with NaN
        - Nested dictionaries -> flattened with prefixed keys
        - Lists of dictionaries -> separate sections
        - Scalar values -> repeated or in separate metadata section
        """
        import pandas as pd
        import numpy as np

        # Check for 2D grid surface data (heterogeneous chi-square landscape)
        if 'chi2_surface' in export_data or 'chi2_grid' in export_data:
            return self._export_grid_surface(export_data)

        # Check for datasets list (merged plots)
        if 'datasets' in export_data and isinstance(export_data['datasets'], list):
            return self._export_datasets_list(export_data)

        # Standard case: dictionary of arrays/scalars
        return self._export_standard_dict(export_data)

    def _export_grid_surface(self, export_data: dict):
        """Export 2D grid surface (chi-square landscape) as flattened CSV."""
        import pandas as pd
        import numpy as np

        # Get grid data - use explicit None checks to avoid numpy truth value errors
        chi2_surface = export_data.get('chi2_surface')
        if chi2_surface is None:
            chi2_surface = export_data.get('chi2_grid')

        tau_T_grid = export_data.get('tau_T_grid')

        tau_w_grid = export_data.get('tau_delta_W_grid')
        if tau_w_grid is None:
            tau_w_grid = export_data.get('tau_w_grid')

        if chi2_surface is None or tau_T_grid is None or tau_w_grid is None:
            # Fallback: try to export what we can
            return self._export_standard_dict(export_data)

        # Handle 2D surface by creating meshgrid and flattening
        if hasattr(chi2_surface, 'ndim') and chi2_surface.ndim == 2:
            rows = []
            for i, tau_T in enumerate(tau_T_grid):
                for j, tau_w in enumerate(tau_w_grid):
                    chi2_val = np.nan
                    if i < chi2_surface.shape[0] and j < chi2_surface.shape[1]:
                        chi2_val = float(chi2_surface[i, j])
                    rows.append({
                        'tau_T_us': float(tau_T),
                        'tau_w_us': float(tau_w),
                        'chi2_reduced': chi2_val
                    })
            df = pd.DataFrame(rows)

            # Add best fit info as comment (first row metadata)
            if 'best_fit' in export_data:
                best = export_data['best_fit']
                logger.info(f"Grid CSV includes {len(rows)} points. Best fit: tau_T={best.get('tau_T', 'N/A')}, tau_w={best.get('tau_w', 'N/A')}")

            return df
        else:
            return self._export_standard_dict(export_data)

    def _export_datasets_list(self, export_data: dict):
        """Export list of datasets (merged plots) with proper padding."""
        import pandas as pd
        import numpy as np

        datasets = export_data['datasets']
        if not datasets:
            return pd.DataFrame()

        # Find maximum length across all datasets
        max_len = 0
        for ds in datasets:
            for key, val in ds.items():
                if isinstance(val, np.ndarray):
                    max_len = max(max_len, len(val))

        # Build flattened dictionary with padded arrays
        flat_data = {}
        for i, ds in enumerate(datasets):
            compound = ds.get('compound', f'dataset_{i+1}')
            ds_type = ds.get('type', 'unknown')
            prefix = f"{compound}_{ds_type}_" if ds_type != 'unknown' else f"{compound}_"

            for key, val in ds.items():
                if key in ('compound', 'type'):
                    continue
                col_name = f"{prefix}{key}"
                if isinstance(val, np.ndarray):
                    # Pad to max length
                    padded = np.full(max_len, np.nan)
                    padded[:len(val)] = val
                    flat_data[col_name] = padded
                elif isinstance(val, (int, float)):
                    # Scalar - repeat
                    flat_data[col_name] = [val] * max_len

        return pd.DataFrame(flat_data)

    def _export_standard_dict(self, export_data: dict):
        """Export standard dictionary with arrays of potentially different lengths."""
        import pandas as pd
        import numpy as np

        # Separate arrays from scalars/nested objects
        array_data = {}
        scalar_data = {}
        nested_data = {}

        for key, val in export_data.items():
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    array_data[key] = val
                else:
                    # Multi-dimensional array - skip or flatten
                    logger.warning(f"Skipping multi-dimensional array '{key}' in CSV export")
            elif isinstance(val, (list, tuple)) and len(val) > 0 and not isinstance(val[0], dict):
                array_data[key] = np.array(val)
            elif isinstance(val, dict):
                nested_data[key] = val
            elif isinstance(val, (int, float, str, bool)) or val is None:
                scalar_data[key] = val
            elif isinstance(val, pd.DataFrame):
                # DataFrame - export directly
                return val

        if not array_data:
            # No arrays - export scalars and nested dicts as single row
            flat_row = {}
            flat_row.update(scalar_data)
            for prefix, nested in nested_data.items():
                for k, v in nested.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        flat_row[f"{prefix}_{k}"] = v
            return pd.DataFrame([flat_row])

        # Find max length and pad all arrays
        max_len = max(len(arr) for arr in array_data.values())

        padded_data = {}
        for key, arr in array_data.items():
            if len(arr) < max_len:
                padded = np.full(max_len, np.nan)
                padded[:len(arr)] = arr
                padded_data[key] = padded
            else:
                padded_data[key] = arr

        # Add scalars as metadata columns (repeated)
        for key, val in scalar_data.items():
            if not isinstance(val, str) or len(val) < 50:  # Don't repeat long strings
                padded_data[key] = [val] * max_len

        # Flatten nested dicts
        for prefix, nested in nested_data.items():
            for k, v in nested.items():
                if isinstance(v, (int, float)):
                    padded_data[f"{prefix}_{k}"] = [v] * max_len

        return pd.DataFrame(padded_data)

    def _export_plot_pdf(self, fig, plot_data: dict):
        """Export plot to PDF."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import os

        try:
            # Prompt for save location
            plot_name = plot_data.get('name', 'plot')
            default_filename = f"{plot_name}.pdf"

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save PDF",
                default_filename,
                "PDF Files (*.pdf)"
            )

            if not file_path:
                return

            # Export to PDF
            fig.savefig(file_path, format='pdf', bbox_inches='tight')

            logger.info(f"Plot exported to PDF: {file_path}")
            self.status_message.emit(f"PDF saved: {os.path.basename(file_path)}")

        except Exception as e:
            logger.error(f"Failed to export PDF: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export PDF:\n{e}"
            )

    def _toggle_log_x(self, fig, canvas, checked: bool):
        """Toggle logarithmic X axis."""
        try:
            # Toggle X scale for all axes
            for ax in fig.get_axes():
                if checked:
                    xmin, xmax = ax.get_xlim()
                    ax.set_xscale('log')
                    ax.set_xlim(0.01, xmax)
                else:
                    ax.set_xscale('linear')
                    ax.autoscale(axis='x')

            # Redraw canvas
            canvas.draw()

            logger.info(f"X axis scale changed to: {'log' if checked else 'linear'}")

        except Exception as e:
            logger.error(f"Failed to toggle log X: {e}")

    # ==================== CLEAR METHODS ====================

    def clear_results(self, clear_plots: bool = False):
        """Clear results and optionally plots."""
        if clear_plots:
            # Close ALL tabs
            self.tabs.clear()
            self.tab_indices = {}
        else:
            # Close only result tabs, preserve plot tabs
            tabs_to_close = []
            result_types = ['Kinetics', 'Quantum Yield', 'Surplus', 'Heterogeneous']

            for i in range(self.tabs.count()):
                tab_name = self.tabs.tabText(i)
                if tab_name in result_types:
                    tabs_to_close.append(i)

            # Close in reverse order
            for index in sorted(tabs_to_close, reverse=True):
                tab_name = self.tabs.tabText(index)
                if tab_name in self.tab_indices:
                    del self.tab_indices[tab_name]
                self.tabs.removeTab(index)

        # Clear result data
        self.kinetics_results = {}
        self.statistics_results = {}
        self.qy_results = []
        self.surplus_results = {}
        self.heterogeneous_results = {}

    def populate_results_from_session(self, analysis_results: Dict):
        """Populate results from session."""
        try:
            if 'kinetics_results' in analysis_results and 'statistics_results' in analysis_results:
                self.populate_kinetics_results(
                    analysis_results['kinetics_results'],
                    analysis_results['statistics_results']
                )

            if 'qy_results' in analysis_results:
                self.populate_qy_results(analysis_results['qy_results'])

            if 'surplus_results' in analysis_results:
                self.populate_surplus_results(analysis_results['surplus_results'])

            if 'heterogeneous_results' in analysis_results:
                self.populate_heterogeneous_results(analysis_results['heterogeneous_results'])

        except Exception as e:
            logger.error(f"Error populating results from session: {e}")
