#!/usr/bin/env python3
"""
Integrated Browser Widget - Origin-style split panel interface

Left panel: Browser with 4 sections (decay_files, abs_files, results, plots)
Right panel: Dynamic content tabs (results with sub-tabs, plots)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QMenu, QHeaderView, QAbstractItemView, QProgressDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QAction
from typing import Dict, List, Optional
import numpy as np

from data.file_parser import FileParser, ParsedFile
from utils.logger_config import get_logger
from gui.dataset_classification_dialog import classify_dataset

logger = get_logger(__name__)


class FileLoadWorker(QThread):
    """Background worker for file loading and parsing."""

    # Signals
    progress = pyqtSignal(str)  # Status message
    finished = pyqtSignal(dict)  # Parsed compounds
    error = pyqtSignal(str)  # Error message

    def __init__(self, folder_path: str, parser: FileParser):
        super().__init__()
        self.folder_path = folder_path
        self.parser = parser

    def run(self):
        """Load and parse files in background."""
        try:
            self.progress.emit(f"Parsing files from: {self.folder_path}...")

            # Parse directory
            compounds = self.parser.parse_directory(self.folder_path)

            # Link absorption data
            for compound_name, files in compounds.items():
                self.progress.emit(f"Processing: {compound_name}...")
                self.parser.link_absorption_data(files, self.folder_path)

            self.progress.emit("File loading complete")
            self.finished.emit(compounds)

        except Exception as e:
            logger.error(f"Error in file load worker: {e}")
            self.error.emit(str(e))


class IntegratedBrowserWidget(QWidget):
    """
    Origin-style browser with split panel.

    Left: Browser tree (decay_files, abs_files, results, plots)
    Right: Dynamic content tabs
    """

    # Signals (keep compatible with existing code)
    status_message = pyqtSignal(str)
    preview_requested = pyqtSignal(str, int, dict)  # compound_name, replicate_num, replicate_data
    plot_requested = pyqtSignal(list)  # Emitted with list of selected items for individual plots
    plot_merged_requested = pyqtSignal(list)  # Emitted for merged plot request
    surplus_plot_requested = pyqtSignal(str)  # Emitted with compound_name for surplus plot
    heterogeneous_plot_requested = pyqtSignal(str)  # Emitted with key for heterogeneous plots
    vesicle_analysis_requested = pyqtSignal(str, int, dict)  # compound_name, replicate_num, replicate_data
    folder_loaded = pyqtSignal(dict)  # Emitted when folder loading completes with compounds dict
    absorption_plot_requested = pyqtSignal(str)  # Emitted with compound_name for single absorption plot
    absorption_merged_requested = pyqtSignal(list)  # Emitted with list of compound_names for merged absorption plot

    # Kinetics tree column indices (checkbox column removed - using highlight selection)
    COL_COMPOUND = 0        # Compound / Replicate name
    COL_WAVELENGTH = 1      # λ (nm)
    COL_CLASSIFICATION = 2  # Dataset classification
    COL_A = 3               # Amplitude (A)
    COL_TAU_DELTA = 4       # τΔ (μs)
    COL_TAU_T = 5           # τT (μs)
    COL_SNR = 6             # SNR
    COL_R_SQUARED = 7       # R²
    COL_CHI_SQUARED = 8     # χ²ᵣ
    COL_MASKED_TIME = 9     # Masked (μs)
    COL_T0 = 10             # t0 (μs)
    COL_Y0 = 11             # y0

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parser = FileParser()
        self.compounds = {}  # Store parsed compounds
        self.current_folder = None

        # Storage for results
        self.kinetics_results = {}
        self.statistics_results = {}
        self.qy_results = []
        self.surplus_results = {}
        self.heterogeneous_results = {}

        # Track which result tabs are open
        self.result_tabs_open = {}  # {'Kinetics': tab_index, 'Quantum Yield': tab_index, ...}

        # Track which plot tabs are open
        self.plot_tabs_open = {}  # {'Kinetics_TMPyP_Mean': tab_index, ...}

        self._setup_ui()

    def _setup_ui(self):
        """Setup split-panel UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT PANEL: Browser tree
        self.browser_tree = self._create_browser_tree()
        splitter.addWidget(self.browser_tree)

        # RIGHT PANEL: Content tabs
        self.content_tabs = QTabWidget()
        self.content_tabs.setTabsClosable(True)
        self.content_tabs.tabCloseRequested.connect(self._close_content_tab)
        splitter.addWidget(self.content_tabs)

        # Set splitter sizes (30% browser, 70% content)
        splitter.setSizes([300, 700])
        splitter.setStretchFactor(0, 0)  # Browser fixed-ish width
        splitter.setStretchFactor(1, 1)  # Content stretches

        layout.addWidget(splitter)

    def _create_browser_tree(self) -> QTreeWidget:
        """Create browser tree with 4 sections."""
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setAlternatingRowColors(True)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Styling - NO hover effects (Session 48 requirement)
        tree.setStyleSheet("""
            QTreeWidget::item:hover {
                background-color: transparent;
            }
            QTreeWidget::item:selected {
                background-color: #FFE4B5;
                color: black;
            }
        """)

        # Context menu
        tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        tree.customContextMenuRequested.connect(self._show_browser_context_menu)

        # Item clicked
        tree.itemClicked.connect(self._on_browser_item_clicked)

        # Create 4 top-level sections
        self._create_browser_sections(tree)

        return tree

    def _create_browser_sections(self, tree: QTreeWidget):
        """Create 4 collapsible sections in browser with improved labels and separators."""

        # Section 1: Decays (was: decay_files)
        self.decay_section = QTreeWidgetItem(tree, ["Decays"])
        self.decay_section.setExpanded(True)
        font = self.decay_section.font(0)
        font.setBold(True)
        self.decay_section.setFont(0, font)
        self.decay_section.setForeground(0, QColor(50, 50, 50))  # Dark gray
        self.decay_section.setData(0, Qt.ItemDataRole.UserRole, {'section': 'decay_files'})

        # Section 2: Abs Spectra (was: abs_files)
        self.abs_section = QTreeWidgetItem(tree, ["Abs Spectra"])
        self.abs_section.setExpanded(True)
        font = self.abs_section.font(0)
        font.setBold(True)
        self.abs_section.setFont(0, font)
        self.abs_section.setForeground(0, QColor(50, 50, 50))  # Dark gray
        self.abs_section.setData(0, Qt.ItemDataRole.UserRole, {'section': 'abs_files'})

        # Separator before results (styled horizontal line)
        spacer1 = QTreeWidgetItem(tree, ["─────────────────────────"])
        spacer1.setFlags(Qt.ItemFlag.NoItemFlags)  # Make unselectable/non-interactive
        spacer1.setForeground(0, QColor(180, 180, 180))  # Light gray
        font_spacer = spacer1.font(0)
        font_spacer.setPointSize(font_spacer.pointSize() - 2)  # Smaller font
        spacer1.setFont(0, font_spacer)

        # Section 3: Results
        self.results_section = QTreeWidgetItem(tree, ["Results"])
        self.results_section.setExpanded(True)
        font = self.results_section.font(0)
        font.setBold(True)
        self.results_section.setFont(0, font)
        self.results_section.setForeground(0, QColor(50, 50, 50))  # Dark gray
        self.results_section.setData(0, Qt.ItemDataRole.UserRole, {'section': 'results'})

        # Separator before plots
        spacer2 = QTreeWidgetItem(tree, ["─────────────────────────"])
        spacer2.setFlags(Qt.ItemFlag.NoItemFlags)  # Make unselectable/non-interactive
        spacer2.setForeground(0, QColor(180, 180, 180))  # Light gray
        font_spacer2 = spacer2.font(0)
        font_spacer2.setPointSize(font_spacer2.pointSize() - 2)  # Smaller font
        spacer2.setFont(0, font_spacer2)

        # Section 4: Plots
        self.plots_section = QTreeWidgetItem(tree, ["Plots"])
        self.plots_section.setExpanded(True)
        font = self.plots_section.font(0)
        font.setBold(True)
        self.plots_section.setFont(0, font)
        self.plots_section.setForeground(0, QColor(50, 50, 50))  # Dark gray
        self.plots_section.setData(0, Qt.ItemDataRole.UserRole, {'section': 'plots'})

    def _on_browser_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle browser item click."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        item_type = data.get('type')

        if item_type == 'result':
            # Open result tab
            result_type = data.get('result_type')
            self._open_result_tab(result_type)

        elif item_type == 'plot':
            # Open plot tab
            plot_name = data.get('plot_name')
            plot_data = data.get('plot_data')
            self._open_plot_tab(plot_name, plot_data)

    def _open_result_tab(self, result_type: str):
        """Open or switch to result tab."""
        # Check if tab already open
        if result_type in self.result_tabs_open:
            tab_index = self.result_tabs_open[result_type]
            # Verify tab still exists
            if tab_index < self.content_tabs.count():
                self.content_tabs.setCurrentIndex(tab_index)
                return

        # Create new result tab
        if result_type == 'Kinetics':
            tab_widget = self._create_kinetics_result_tab()
        elif result_type == 'Quantum Yield':
            tab_widget = self._create_qy_result_tab()
        elif result_type == 'Surplus':
            tab_widget = self._create_surplus_result_tab()
        elif result_type == 'Heterogeneous':
            tab_widget = self._create_heterogeneous_result_tab()
        else:
            logger.warning(f"Unknown result type: {result_type}")
            return

        # Add tab
        tab_index = self.content_tabs.addTab(tab_widget, result_type)
        self.content_tabs.setCurrentIndex(tab_index)
        self.result_tabs_open[result_type] = tab_index

    def _open_plot_tab(self, plot_name: str, plot_data: dict):
        """Open or switch to plot tab."""
        # Check if tab already open
        if plot_name in self.plot_tabs_open:
            tab_index = self.plot_tabs_open[plot_name]
            # Verify tab still exists
            if tab_index < self.content_tabs.count():
                self.content_tabs.setCurrentIndex(tab_index)
                return

        # Create new plot tab
        tab_widget = self._create_plot_tab_widget(plot_data)

        # Add tab
        tab_index = self.content_tabs.addTab(tab_widget, plot_name)
        self.content_tabs.setCurrentIndex(tab_index)
        self.plot_tabs_open[plot_name] = tab_index

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

                    # Check current X-axis scale and set button state accordingly
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

                # Store references for later use
                widget._canvas = canvas
                widget._figure = fig
                widget._plot_data = plot_data

            except Exception as e:
                logger.error(f"Failed to embed matplotlib figure: {e}")
                label = QLabel(f"Error embedding plot: {e}")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
        else:
            # Fallback for missing figure
            label = QLabel(f"Plot: {plot_data.get('name', 'Unknown')}\n(No figure data)")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

        return widget

    def _create_kinetics_result_tab(self) -> QWidget:
        """Create Kinetics results tab (summary tree only)."""
        # Return summary widget directly (no sub-tabs)
        summary_widget = self._create_kinetics_summary_widget()
        return summary_widget

    def _create_kinetics_summary_widget(self) -> QWidget:
        """Create kinetics summary tree widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create tree with ExtendedSelection (Ctrl+Click, Shift+Click)
        tree = QTreeWidget()
        tree.setColumnCount(12)  # Removed checkbox column
        tree.setHeaderLabels([
            "Compound / Replicate", "λ (nm)", "Classification",
            "A", "τΔ (μs)", "τT (μs)", "SNR", "R²", "χ²ᵣ", "Masked (μs)", "t0 (μs)", "y0"
        ])
        tree.setAlternatingRowColors(True)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        tree.setStyleSheet("""
            QTreeWidget::item:hover {
                background-color: transparent;
            }
            QTreeWidget::item:selected {
                background-color: #FFE4B5;
                color: black;
            }
        """)

        # Auto-resize columns to content
        header = tree.header()
        for col in range(12):  # Updated for 12 columns
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        # Store reference for population
        self.kinetics_tree = tree

        layout.addWidget(tree)

        # Populate tree if data already exists
        if self.kinetics_results and self.statistics_results:
            self._populate_kinetics_tree()

        return widget

    def _create_qy_result_tab(self) -> QWidget:
        """Create Quantum Yield results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create table widget
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
        self.qy_table = QTableWidget()
        self.qy_table.setColumnCount(6)
        self.qy_table.setHorizontalHeaderLabels([
            "Sample", "Standard", "λ (nm)", "Φ", "Error", "Rel. Error (%)"
        ])
        self.qy_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.qy_table.setAlternatingRowColors(True)
        self.qy_table.verticalHeader().setVisible(False)

        # LEFT-align all header labels
        for col in range(6):
            item = self.qy_table.horizontalHeaderItem(col)
            if item:
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Set selection color, disable hover
        self.qy_table.setStyleSheet("""
            QTableWidget::item:hover {
                background-color: transparent;
            }
            QTableWidget::item:selected {
                background-color: #FFE4B5;
                color: black;
            }
        """)

        # Auto-resize columns
        header = self.qy_table.horizontalHeader()
        for col in range(6):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.qy_table)

        # Populate if data exists
        if self.qy_results:
            self._populate_qy_table()

        return widget

    def _create_surplus_result_tab(self) -> QWidget:
        """Create Surplus results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create table widget
        from PyQt6.QtWidgets import QTableWidget
        self.surplus_table = QTableWidget()
        self.surplus_table.setColumnCount(8)
        self.surplus_table.setHorizontalHeaderLabels([
            "Compound", "α", "β", "τΔ,1 (μs)", "τΔ,2 (μs)", "τT (μs)", "R²", "Mask (μs)"
        ])
        self.surplus_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.surplus_table.setAlternatingRowColors(True)
        self.surplus_table.verticalHeader().setVisible(False)

        # LEFT-align all header labels
        for col in range(8):
            item = self.surplus_table.horizontalHeaderItem(col)
            if item:
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Set selection color, disable hover
        self.surplus_table.setStyleSheet("""
            QTableWidget::item:hover {
                background-color: transparent;
            }
            QTableWidget::item:selected {
                background-color: #FFE4B5;
                color: black;
            }
        """)

        # Auto-resize columns
        header = self.surplus_table.horizontalHeader()
        for col in range(9):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.surplus_table)

        # Populate if data exists
        if self.surplus_results:
            self._populate_surplus_table()

        return widget

    def _create_heterogeneous_result_tab(self) -> QWidget:
        """Create Heterogeneous results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create table widget
        from PyQt6.QtWidgets import QTableWidget
        self.heterogeneous_table = QTableWidget()
        self.heterogeneous_table.setColumnCount(11)
        self.heterogeneous_table.setHorizontalHeaderLabels([
            "Compound", "Rep", "τT (μs)", "τw (μs)", "τL (μs)", "A", "B", "C", "A/B", "χ²red", "Action"
        ])
        self.heterogeneous_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.heterogeneous_table.setAlternatingRowColors(True)
        self.heterogeneous_table.verticalHeader().setVisible(False)

        # LEFT-align all header labels
        for col in range(11):
            item = self.heterogeneous_table.horizontalHeaderItem(col)
            if item:
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Set selection color, disable hover
        self.heterogeneous_table.setStyleSheet("""
            QTableWidget::item:hover {
                background-color: transparent;
            }
            QTableWidget::item:selected {
                background-color: #FFE4B5;
                color: black;
            }
        """)

        # Auto-resize columns
        header = self.heterogeneous_table.horizontalHeader()
        for col in range(11):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.heterogeneous_table)

        # Populate if data exists
        if self.heterogeneous_results:
            self._populate_heterogeneous_table()

        return widget

    def _close_content_tab(self, index: int):
        """Handle content tab close."""
        # Get tab name
        tab_name = self.content_tabs.tabText(index)

        # Remove from tracking dicts and update indices
        if tab_name in self.result_tabs_open:
            del self.result_tabs_open[tab_name]

        if tab_name in self.plot_tabs_open:
            del self.plot_tabs_open[tab_name]

        # Remove tab first
        self.content_tabs.removeTab(index)

        # Update ALL indices for tabs that were after the closed one
        for key in list(self.result_tabs_open.keys()):
            if self.result_tabs_open[key] > index:
                self.result_tabs_open[key] -= 1

        for key in list(self.plot_tabs_open.keys()):
            if self.plot_tabs_open[key] > index:
                self.plot_tabs_open[key] -= 1

    def _get_selected_absorption_compounds(self) -> List[str]:
        """Get list of selected (highlighted) absorption compound names."""
        selected = []

        # Get selected items from browser tree
        selected_items = self.browser_tree.selectedItems()

        for item in selected_items:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get('type') == 'abs_compound':
                selected.append(data.get('compound'))

        return selected

    def _show_browser_context_menu(self, position):
        """Show context menu for browser items."""
        item = self.browser_tree.itemAt(position)
        if not item:
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        menu = QMenu(self)

        # Different menus based on item type
        if data.get('type') == 'decay_compound':
            compound_name = data.get('compound')
            decay_files = data.get('files', [])

            # Add preview submenu for each replicate
            if decay_files:
                preview_menu = menu.addMenu("Preview")
                for idx, decay_file in enumerate(decay_files):
                    # Get replicate data
                    time, intensity_list = decay_file.get_kinetics_data()
                    for rep_idx, intensity in enumerate(intensity_list):
                        rep_num = rep_idx + 1
                        replicate_data = {
                            'time': time,
                            'intensity': intensity,
                            'file_path': decay_file.file_path
                        }

                        # Create action for this replicate
                        action = QAction(f"Replicate {rep_num}", self)
                        action.triggered.connect(
                            lambda checked=False, cn=compound_name, rn=rep_num, rd=replicate_data:
                            self.preview_requested.emit(cn, rn, rd)
                        )
                        preview_menu.addAction(action)

        elif data.get('type') == 'abs_compound':
            compound_name = data.get('compound')

            # Add "Plot Spectrum" action
            plot_action = QAction("Plot Spectrum", self)
            plot_action.triggered.connect(
                lambda checked=False, cn=compound_name:
                self.absorption_plot_requested.emit(cn)
            )
            menu.addAction(plot_action)

            # Check if multiple absorption compounds are selected
            selected_abs_compounds = self._get_selected_absorption_compounds()
            if len(selected_abs_compounds) > 1:
                # Add "Plot Merged Spectra" action
                merged_action = QAction(f"Plot Merged Spectra ({len(selected_abs_compounds)} compounds)", self)
                merged_action.triggered.connect(
                    lambda checked=False, compounds=selected_abs_compounds:
                    self.absorption_merged_requested.emit(compounds)
                )
                menu.addAction(merged_action)

        if menu.actions():
            menu.exec(self.browser_tree.viewport().mapToGlobal(position))

    # ==================== DATA LOADING ====================

    def load_folder(self, folder_path: str):
        """Load and parse data files from folder (using background thread)."""
        self.current_folder = folder_path

        # Create progress dialog
        self.progress_dialog = QProgressDialog("Loading files...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Loading Data")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.canceled.connect(self._cancel_file_loading)

        # Create and start worker thread
        self.file_load_worker = FileLoadWorker(folder_path, self.parser)
        self.file_load_worker.progress.connect(self._on_load_progress)
        self.file_load_worker.finished.connect(self._on_load_finished)
        self.file_load_worker.error.connect(self._on_load_error)
        self.file_load_worker.start()

    def _cancel_file_loading(self):
        """Cancel file loading operation."""
        if hasattr(self, 'file_load_worker') and self.file_load_worker.isRunning():
            self.file_load_worker.terminate()
            self.file_load_worker.wait()
            self.status_message.emit("File loading canceled")
            logger.info("File loading canceled by user")

    def _on_load_progress(self, message: str):
        """Update progress dialog with status."""
        self.progress_dialog.setLabelText(message)
        self.status_message.emit(message)

    def _on_load_finished(self, compounds: dict):
        """Handle completion of file loading."""
        try:
            self.compounds = compounds
            self.progress_dialog.close()

            # Classify datasets (still needs to be on main thread for dialogs)
            self._classify_datasets()

            # Populate browser
            self._populate_decay_section()
            self._populate_abs_section()

            # Update status
            n_compounds = len(self.compounds)
            total_replicates = sum(len([f for f in files if f.file_type == 'decay'])
                                  for files in self.compounds.values())
            self.status_message.emit(
                f"Loaded {n_compounds} compounds ({total_replicates} decay files)"
            )

            logger.info(f"Loaded {n_compounds} compounds from {self.current_folder}")

            # Emit signal that folder loading is complete (for solis_gui.py)
            self.folder_loaded.emit(self.compounds)

        except Exception as e:
            logger.error(f"Error processing loaded files: {e}")
            self.status_message.emit(f"Error processing files: {e}")

    def _on_load_error(self, error_message: str):
        """Handle file loading error."""
        self.progress_dialog.close()
        self.status_message.emit(f"Error loading folder: {error_message}")
        logger.error(f"File loading error: {error_message}")

    def _classify_datasets(self):
        """Ask user to classify dataset structure for each compound."""
        try:
            global_dataset_type = None

            for compound_name in sorted(self.compounds.keys()):
                files = self.compounds[compound_name]
                decay_files = [f for f in files if f.file_type == 'decay']
                if not decay_files:
                    continue

                # Check if already classified
                if decay_files[0].dataset_type is not None:
                    continue

                # Apply global classification if set
                if global_dataset_type is not None:
                    for f in decay_files:
                        f.dataset_type = global_dataset_type
                    continue

                # Get first replicate for preview
                first_file = decay_files[0]
                try:
                    time_data, intensity_replicates = first_file.get_kinetics_data()
                    first_intensity = intensity_replicates[0]
                except Exception as e:
                    logger.error(f"Failed to load data for {compound_name}: {e}")
                    continue

                # Show classification dialog
                self.status_message.emit(f"Classifying dataset: {compound_name}...")
                dataset_type, apply_to_all = classify_dataset(
                    compound_name, time_data, first_intensity, parent=self
                )

                if dataset_type is None:
                    dataset_type = 'auto'

                # Apply classification
                for f in decay_files:
                    f.dataset_type = dataset_type

                if apply_to_all:
                    global_dataset_type = dataset_type

            self.status_message.emit("Dataset classification complete")

        except Exception as e:
            logger.error(f"Error during dataset classification: {e}")
            self.status_message.emit(f"Classification error: {e}")

    def _populate_decay_section(self):
        """Populate decay_files section.

        OPTIMIZED: Disables updates during batch operations for 5-10x faster population.
        """
        # Disable updates during population (OPTIMIZATION)
        self.browser_tree.setUpdatesEnabled(False)

        try:
            # Clear existing
            self.decay_section.takeChildren()

            for compound_name in sorted(self.compounds.keys()):
                files = self.compounds[compound_name]
                decay_files = [f for f in files if f.file_type == 'decay']

                if not decay_files:
                    continue

                # Get metadata from first file
                first_decay = decay_files[0]
                wavelength = first_decay.wavelength if hasattr(first_decay, 'wavelength') else '—'

                # Count replicates
                n_replicates = sum(len(f.get_kinetics_data()[1]) for f in decay_files)

                # Create item: "Compound [λ nm, N=count]"
                if isinstance(wavelength, (int, float)):
                    label = f"{compound_name} [{wavelength:.0f} nm, N={n_replicates}]"
                else:
                    label = f"{compound_name} [N={n_replicates}]"

                item = QTreeWidgetItem(self.decay_section, [label])

                # Store data
                item.setData(0, Qt.ItemDataRole.UserRole, {
                    'type': 'decay_compound',
                    'compound': compound_name,
                    'files': decay_files
                })

        finally:
            # Re-enable updates and refresh (OPTIMIZATION)
            self.browser_tree.setUpdatesEnabled(True)

    def _populate_abs_section(self):
        """Populate abs_files section.

        OPTIMIZED: Disables updates during batch operations for 5-10x faster population.
        """
        # Disable updates during population (OPTIMIZATION)
        self.browser_tree.setUpdatesEnabled(False)

        try:
            # Clear existing
            self.abs_section.takeChildren()

            # Collect unique compounds with abs files
            abs_compounds = set()
            for compound_name, files in self.compounds.items():
                abs_files = [f for f in files if f.file_type == 'absorption']
                if abs_files:
                    abs_compounds.add(compound_name)

            for compound_name in sorted(abs_compounds):
                item = QTreeWidgetItem(self.abs_section, [compound_name])

                # Store data
                item.setData(0, Qt.ItemDataRole.UserRole, {
                    'type': 'abs_compound',
                    'compound': compound_name
                })

        finally:
            # Re-enable updates and refresh (OPTIMIZATION)
            self.browser_tree.setUpdatesEnabled(True)

    # ==================== RESULTS POPULATION ====================

    def populate_kinetics_results(self, kinetics_results: Dict, statistics_results: Dict):
        """Populate kinetics results."""
        # Store data
        self.kinetics_results = kinetics_results
        self.statistics_results = statistics_results

        # Add to results section if not already there
        self._add_result_item('Kinetics')

        # If kinetics tab is already open, populate it
        # (This will happen after the tab is opened and kinetics_tree exists)
        if hasattr(self, 'kinetics_tree') and self.kinetics_tree is not None:
            self._populate_kinetics_tree()

    def _populate_kinetics_tree(self):
        """Populate the kinetics tree widget.

        OPTIMIZED: Disables updates during batch operations for 5-10x faster population.
        """
        # Disable updates during population (OPTIMIZATION)
        self.kinetics_tree.setUpdatesEnabled(False)

        try:
            self.kinetics_tree.clear()

            for compound_name in sorted(self.kinetics_results.keys()):
                compound_data = self.kinetics_results[compound_name]
                results_list = compound_data['results']
                wavelength = compound_data.get('wavelength', '—')
                classification = compound_data.get('classification', '—')
                stats_data = self.statistics_results.get(compound_name)

                # Create parent item (compound with mean ± SD)
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
            # Re-enable updates and refresh (OPTIMIZATION)
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
        for col in range(12):  # Updated for 12 columns
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

    def _add_result_item(self, result_type: str):
        """Add result item to results section."""
        # Check if already exists
        for i in range(self.results_section.childCount()):
            child = self.results_section.child(i)
            if child.text(0) == result_type:
                return  # Already exists

        # Create new item
        item = QTreeWidgetItem(self.results_section, [result_type])
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'result',
            'result_type': result_type
        })

    def _add_plot_item(self, plot_name: str, plot_data: dict):
        """Add plot item to plots section."""
        # Check if already exists
        for i in range(self.plots_section.childCount()):
            child = self.plots_section.child(i)
            if child.text(0) == plot_name:
                return  # Already exists

        # Create new item
        item = QTreeWidgetItem(self.plots_section, [plot_name])
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'plot',
            'plot_name': plot_name,
            'plot_data': plot_data
        })

    # ==================== COMPATIBILITY METHODS ====================

    def _get_selected_items_from_tree(self, tree: QTreeWidget, include_children: bool = True) -> List:
        """
        Get selected items from tree (highlight-based selection).

        Args:
            tree: QTreeWidget to get selections from
            include_children: Not used anymore (kept for compatibility)

        Returns:
            List of QTreeWidgetItem that are selected (highlighted)
        """
        # Simply return the selected items from Qt's selection model
        return tree.selectedItems()

    def get_checked_compounds(self) -> Dict[str, List[dict]]:
        """Get selected compounds from decay_files section (highlight-based)."""
        selected = {}

        # Get selected items from browser tree
        selected_items = self.browser_tree.selectedItems()

        for item in selected_items:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get('type') == 'decay_compound':
                compound_name = data['compound']
                decay_files = data['files']

                # Get all replicates
                replicates = []
                for decay_file in decay_files:
                    time, intensity_replicates = decay_file.get_kinetics_data()
                    for rep_idx, intensity in enumerate(intensity_replicates):
                        replicates.append({
                            'decay_file': decay_file,
                            'replicate_index': rep_idx,
                            'time': time,
                            'intensity': intensity
                        })

                if replicates:
                    selected[compound_name] = replicates

        return selected

    def get_selected_compounds_for_analysis(self) -> Dict[str, List]:
        """Get selected compounds for analysis (highlight-based)."""
        selected = {}

        # Get selected items from browser tree
        selected_items = self.browser_tree.selectedItems()

        for item in selected_items:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get('type') == 'decay_compound':
                compound_name = data['compound']
                if compound_name in self.compounds:
                    decay_files = [f for f in self.compounds[compound_name]
                                 if f.file_type == 'decay']
                    if decay_files:
                        selected[compound_name] = decay_files

        return selected

    def _on_view_plot(self):
        """Handle Individual Plots button click."""
        selected_items = []

        # Check if kinetics tree exists and has data
        if not hasattr(self, 'kinetics_tree') or self.kinetics_tree is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Data",
                              "Please run kinetics analysis first to view plots.")
            return

        # Check if there are any results
        if not self.kinetics_results or not self.statistics_results:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Results",
                              "No analysis results available. Please run analysis first.")
            return

        # Get selected items from kinetics tree (highlight-based)
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

        if selected_items:
            self.plot_requested.emit(selected_items)
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Selection",
                              "Please select at least one item to plot.")

    def _on_plot_merged(self):
        """Handle Plot Merged button click."""
        selected_items = []

        # Check if kinetics tree exists and has data
        if not hasattr(self, 'kinetics_tree') or self.kinetics_tree is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Data",
                              "Please run kinetics analysis first to view plots.")
            return

        # Check if there are any results
        if not self.kinetics_results or not self.statistics_results:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Results",
                              "No analysis results available. Please run analysis first.")
            return

        # Get selected items from kinetics tree (highlight-based)
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

        if selected_items:
            self.plot_merged_requested.emit(selected_items)
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Selection",
                              "Please select at least one item to plot.")

    def clear_tree(self):
        """Clear decay files tree."""
        self.decay_section.takeChildren()
        self.abs_section.takeChildren()
        self.compounds = {}

    def clear_data(self):
        """Clear all loaded data and reset browser."""
        self.clear_tree()
        self.current_folder = None

    def clear_results(self, clear_plots: bool = False):
        """
        Clear all results and optionally plots.

        Parameters
        ----------
        clear_plots : bool, optional
            If True, also clear plots section and close plot tabs.
            If False (default), preserve plots across analysis runs.
        """
        self.results_section.takeChildren()

        if clear_plots:
            # Full reset: clear plots section too
            self.plots_section.takeChildren()

            # Close ALL tabs (results and plots)
            self.content_tabs.clear()
            self.result_tabs_open = {}
            self.plot_tabs_open = {}
        else:
            # Analysis re-run: preserve plots
            # Close only result tabs, not plot tabs
            tabs_to_close = []
            for i in range(self.content_tabs.count()):
                tab_name = self.content_tabs.tabText(i)
                if tab_name in self.result_tabs_open:
                    tabs_to_close.append(i)

            # Close in reverse order to avoid index shifting
            for index in sorted(tabs_to_close, reverse=True):
                self.content_tabs.removeTab(index)

            # Clear result tabs tracking but preserve plot_tabs_open
            self.result_tabs_open = {}

        # Clear result data
        self.kinetics_results = {}
        self.statistics_results = {}
        self.qy_results = []
        self.surplus_results = {}
        self.heterogeneous_results = {}

    # Stub methods for compatibility
    def populate_qy_results(self, qy_results: List):
        """Populate QY results."""
        self.qy_results = qy_results
        if qy_results:
            self._add_result_item('Quantum Yield')
            # If QY tab is open, populate it
            if 'Quantum Yield' in self.result_tabs_open and hasattr(self, 'qy_table'):
                self._populate_qy_table()

    def _populate_qy_table(self):
        """Populate the QY table widget."""
        from PyQt6.QtWidgets import QTableWidgetItem
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
        """Populate surplus results and auto-generate plots."""
        self.surplus_results = surplus_results
        if surplus_results:
            self._add_result_item('Surplus')
            # If Surplus tab is open, populate it
            if 'Surplus' in self.result_tabs_open and hasattr(self, 'surplus_table'):
                self._populate_surplus_table()

            # Auto-generate plots for all surplus results
            for compound_name in surplus_results.keys():
                self.surplus_plot_requested.emit(compound_name)

    def _populate_surplus_table(self):
        """Populate the Surplus table widget."""
        from PyQt6.QtWidgets import QTableWidgetItem, QPushButton
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
        if heterogeneous_results:
            self._add_result_item('Heterogeneous')
            # If Heterogeneous tab is open, populate it
            if 'Heterogeneous' in self.result_tabs_open and hasattr(self, 'heterogeneous_table'):
                self._populate_heterogeneous_table()

    def _populate_heterogeneous_table(self):
        """Populate the Heterogeneous table widget."""
        from PyQt6.QtWidgets import QTableWidgetItem, QPushButton
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

            # Fit parameters (from dataclass)
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

    def populate_from_session(self, loaded_compounds: Dict, folder_path: str = None):
        """Populate browser from session."""
        try:
            self.compounds = loaded_compounds
            if folder_path:
                self.current_folder = folder_path

            self._populate_decay_section()
            self._populate_abs_section()

            n_compounds = len(self.compounds)
            total_replicates = sum(len([f for f in files if f.file_type == 'decay'])
                                  for files in self.compounds.values())
            self.status_message.emit(
                f"Session loaded: {n_compounds} compounds ({total_replicates} decay files)"
            )

        except Exception as e:
            logger.error(f"Error populating browser from session: {e}")

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

    def _export_plot_csv(self, fig, plot_data: dict):
        """Export plot data to CSV with robust handling of various data structures."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import pandas as pd
        import numpy as np
        import os

        try:
            # Check if figure has export data attached
            if not hasattr(fig, 'solis_export_data'):
                logger.warning("No export data attached to figure")
                QMessageBox.warning(
                    self,
                    "Export Error",
                    "This plot does not have exportable data attached."
                )
                return

            # Get export data (handle both callable and dict for backward compatibility)
            export_data_attr = fig.solis_export_data
            if callable(export_data_attr):
                export_data = export_data_attr()
            else:
                export_data = export_data_attr

            # Prompt user for save location
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
            return self._export_standard_dict(export_data)

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
            return pd.DataFrame(rows)
        else:
            return self._export_standard_dict(export_data)

    def _export_datasets_list(self, export_data: dict):
        """Export list of datasets (merged plots) with proper padding."""
        import pandas as pd
        import numpy as np

        datasets = export_data['datasets']
        if not datasets:
            return pd.DataFrame()

        max_len = 0
        for ds in datasets:
            for key, val in ds.items():
                if isinstance(val, np.ndarray):
                    max_len = max(max_len, len(val))

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
                    padded = np.full(max_len, np.nan)
                    padded[:len(val)] = val
                    flat_data[col_name] = padded
                elif isinstance(val, (int, float)):
                    flat_data[col_name] = [val] * max_len

        return pd.DataFrame(flat_data)

    def _export_standard_dict(self, export_data: dict):
        """Export standard dictionary with arrays of potentially different lengths."""
        import pandas as pd
        import numpy as np

        array_data = {}
        scalar_data = {}
        nested_data = {}

        for key, val in export_data.items():
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    array_data[key] = val
                else:
                    logger.warning(f"Skipping multi-dimensional array '{key}' in CSV export")
            elif isinstance(val, (list, tuple)) and len(val) > 0 and not isinstance(val[0], dict):
                array_data[key] = np.array(val)
            elif isinstance(val, dict):
                nested_data[key] = val
            elif isinstance(val, (int, float, str, bool)) or val is None:
                scalar_data[key] = val
            elif isinstance(val, pd.DataFrame):
                return val

        if not array_data:
            flat_row = {}
            flat_row.update(scalar_data)
            for prefix, nested in nested_data.items():
                for k, v in nested.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        flat_row[f"{prefix}_{k}"] = v
            return pd.DataFrame([flat_row])

        max_len = max(len(arr) for arr in array_data.values())

        padded_data = {}
        for key, arr in array_data.items():
            if len(arr) < max_len:
                padded = np.full(max_len, np.nan)
                padded[:len(arr)] = arr
                padded_data[key] = padded
            else:
                padded_data[key] = arr

        for key, val in scalar_data.items():
            if not isinstance(val, str) or len(val) < 50:
                padded_data[key] = [val] * max_len

        for prefix, nested in nested_data.items():
            for k, v in nested.items():
                if isinstance(v, (int, float)):
                    padded_data[f"{prefix}_{k}"] = [v] * max_len

        return pd.DataFrame(padded_data)

    def _export_plot_pdf(self, fig, plot_data: dict):
        """Export plot to PDF."""
        from PyQt6.QtWidgets import QFileDialog
        import os

        try:
            # Prompt user for save location
            plot_name = plot_data.get('name', 'plot')
            default_filename = f"{plot_name}.pdf"

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save PDF",
                default_filename,
                "PDF Files (*.pdf)"
            )

            if not file_path:
                return  # User cancelled

            # Export to PDF
            fig.savefig(file_path, format='pdf', bbox_inches='tight')

            logger.info(f"Plot exported to PDF: {file_path}")
            self.status_message.emit(f"PDF saved: {os.path.basename(file_path)}")

        except Exception as e:
            logger.error(f"Failed to export PDF: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export PDF:\n{e}"
            )

    def _toggle_log_x(self, fig, canvas, checked: bool):
        """Toggle logarithmic X axis."""
        try:
            # Toggle X scale for all axes in figure
            for ax in fig.get_axes():
                if checked:
                    # Get current x limits
                    xmin, xmax = ax.get_xlim()
                    # Set log scale
                    ax.set_xscale('log')
                    # Set limits from 0.01 to xmax (or slightly more)
                    ax.set_xlim(0.01, xmax)
                else:
                    ax.set_xscale('linear')
                    # Restore to auto limits
                    ax.autoscale(axis='x')

            # Redraw canvas
            canvas.draw()

            logger.info(f"X axis scale changed to: {'log' if checked else 'linear'}")

        except Exception as e:
            logger.error(f"Failed to toggle log X: {e}")
