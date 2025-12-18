#!/usr/bin/env python3
"""
File Browser Widget - Left panel with file/results/plots tree

Shows 4 sections:
- Decays (decay files)
- Abs Spectra (absorption files)
- Results (kinetics, QY, surplus, heterogeneous)
- Plots (generated plots)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMenu, QAbstractItemView, QProgressDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QFont, QColor, QAction, QIcon
from typing import Dict, List
from pathlib import Path

from data.file_parser import FileParser, ParsedFile
from utils.logger_config import get_logger
from gui.dataset_classification_dialog import classify_dataset
from gui.stylesheets import TREE_STYLE

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


class FileBrowserWidget(QWidget):
    """
    File browser tree widget - LEFT PANEL

    Shows 4 sections:
    - Decays (decay files)
    - Abs Spectra (absorption files)
    - Results (kinetics, QY, surplus, heterogeneous)
    - Plots (generated plots)
    """

    # Signals
    status_message = pyqtSignal(str)
    preview_requested = pyqtSignal(str, int, dict)  # compound_name, replicate_num, replicate_data
    folder_loaded = pyqtSignal(dict)  # Emitted when folder loading completes with compounds dict
    absorption_plot_requested = pyqtSignal(str)  # Emitted with compound_name for single absorption plot
    absorption_merged_requested = pyqtSignal(list)  # Emitted with list of compound_names for merged absorption plot
    result_item_clicked = pyqtSignal(str)  # result_type (e.g., 'Kinetics', 'Quantum Yield')
    plot_item_clicked = pyqtSignal(str, dict)  # plot_name, plot_data

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parser = FileParser()
        self.compounds = {}  # Store parsed compounds
        self.current_folder = None

        # Load icons
        self._load_icons()

        self._setup_ui()

    def _load_icons(self):
        """Load SVG icons for browser items."""
        icon_dir = Path(__file__).parent.parent / 'icons'

        self.icon_decay = QIcon(str(icon_dir / 'decay2.svg'))
        self.icon_abs = QIcon(str(icon_dir / 'ABS2.svg'))
        self.icon_result = QIcon(str(icon_dir / 'results2.svg'))
        self.icon_plot = QIcon(str(icon_dir / 'plots2.svg'))

    def _setup_ui(self):
        """Setup UI with tree widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # BROWSER label (matching RESULTS label on right side)
        from PyQt6.QtWidgets import QLabel
        browser_header = QLabel("BROWSER")
        browser_header.setStyleSheet("color: rgb(120, 120, 120); font-weight: bold; padding-left: 5px; padding-bottom: 1px;")
        font = browser_header.font()
        font.setPointSize(font.pointSize() - 1)  # Slightly smaller
        browser_header.setFont(font)
        layout.addWidget(browser_header)

        # Create browser tree
        self.tree = self._create_browser_tree()
        layout.addWidget(self.tree)

    def _create_browser_tree(self) -> QTreeWidget:
        """Create browser tree with 4 sections."""
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setAlternatingRowColors(False)  # CRITICAL: Disable to prevent platform hover effects
        tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        tree.setRootIsDecorated(False)  # Hide expand/collapse arrows - sections always visible

        # Simple styling - selection only, NO hover effects
        tree.setStyleSheet(TREE_STYLE)

        # Context menu
        tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        tree.customContextMenuRequested.connect(self._show_context_menu)

        # Item clicked
        tree.itemClicked.connect(self._on_item_clicked)

        # Create 4 top-level sections
        self._create_sections(tree)

        return tree

    def _create_sections(self, tree: QTreeWidget):
        """Create 4 collapsible sections in browser."""

        # Section 1: Decays
        self.decay_section = QTreeWidgetItem(tree, ["Decays"])
        self.decay_section.setExpanded(True)
        self.decay_section.setFlags(Qt.ItemFlag.ItemIsEnabled)  # Non-selectable header
        # NO setBackground() - let stylesheet handle all backgrounds
        font = self.decay_section.font(0)
        font.setBold(True)
        self.decay_section.setFont(0, font)
        self.decay_section.setForeground(0, QColor(50, 50, 50))
        self.decay_section.setData(0, Qt.ItemDataRole.UserRole, {'section': 'decay_files'})

        # Section 2: Abs Spectra
        self.abs_section = QTreeWidgetItem(tree, ["Abs Spectra"])
        self.abs_section.setExpanded(True)
        self.abs_section.setFlags(Qt.ItemFlag.ItemIsEnabled)  # Non-selectable header
        # NO setBackground() - let stylesheet handle all backgrounds
        font = self.abs_section.font(0)
        font.setBold(True)
        self.abs_section.setFont(0, font)
        self.abs_section.setForeground(0, QColor(50, 50, 50))
        self.abs_section.setData(0, Qt.ItemDataRole.UserRole, {'section': 'abs_files'})

        # Separator
        spacer1 = QTreeWidgetItem(tree, ["─────────────────────────"])
        spacer1.setFlags(Qt.ItemFlag.NoItemFlags)
        # NO setBackground() - let stylesheet handle all backgrounds
        spacer1.setForeground(0, QColor(180, 180, 180))
        font_spacer = spacer1.font(0)
        font_spacer.setPointSize(font_spacer.pointSize() - 2)
        spacer1.setFont(0, font_spacer)

        # Section 3: Results
        self.results_section = QTreeWidgetItem(tree, ["Results"])
        self.results_section.setExpanded(True)
        self.results_section.setFlags(Qt.ItemFlag.ItemIsEnabled)  # Non-selectable header
        # NO setBackground() - let stylesheet handle all backgrounds
        font = self.results_section.font(0)
        font.setBold(True)
        self.results_section.setFont(0, font)
        self.results_section.setForeground(0, QColor(50, 50, 50))
        self.results_section.setData(0, Qt.ItemDataRole.UserRole, {'section': 'results'})

        # Separator
        spacer2 = QTreeWidgetItem(tree, ["─────────────────────────"])
        spacer2.setFlags(Qt.ItemFlag.NoItemFlags)
        # NO setBackground() - let stylesheet handle all backgrounds
        spacer2.setForeground(0, QColor(180, 180, 180))
        font_spacer2 = spacer2.font(0)
        font_spacer2.setPointSize(font_spacer2.pointSize() - 2)
        spacer2.setFont(0, font_spacer2)

        # Section 4: Plots
        self.plots_section = QTreeWidgetItem(tree, ["Plots"])
        self.plots_section.setExpanded(True)
        self.plots_section.setFlags(Qt.ItemFlag.ItemIsEnabled)  # Non-selectable header
        # NO setBackground() - let stylesheet handle all backgrounds
        font = self.plots_section.font(0)
        font.setBold(True)
        self.plots_section.setFont(0, font)
        self.plots_section.setForeground(0, QColor(50, 50, 50))
        self.plots_section.setData(0, Qt.ItemDataRole.UserRole, {'section': 'plots'})

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click - emit signals for results/plots."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        item_type = data.get('type')

        if item_type == 'result':
            # Emit signal for result click
            result_type = data.get('result_type')
            self.result_item_clicked.emit(result_type)

        elif item_type == 'plot':
            # Emit signal for plot click
            plot_name = data.get('plot_name')
            plot_data = data.get('plot_data')
            self.plot_item_clicked.emit(plot_name, plot_data)

    def _show_context_menu(self, position):
        """Show context menu for browser items."""
        item = self.tree.itemAt(position)
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
            menu.exec(self.tree.viewport().mapToGlobal(position))

    def _get_selected_absorption_compounds(self) -> List[str]:
        """Get list of selected (highlighted) absorption compound names."""
        selected = []

        # Get selected items from tree
        selected_items = self.tree.selectedItems()

        for item in selected_items:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get('type') == 'abs_compound':
                selected.append(data.get('compound'))

        return selected

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

            # Classify datasets
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

            # Emit signal that folder loading is complete
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
        """Populate decay section."""
        self.tree.setUpdatesEnabled(False)

        try:
            self.decay_section.takeChildren()

            for compound_name in sorted(self.compounds.keys()):
                files = self.compounds[compound_name]
                decay_files = [f for f in files if f.file_type == 'decay']

                if not decay_files:
                    continue

                # Get metadata
                first_decay = decay_files[0]
                wavelength = first_decay.wavelength if hasattr(first_decay, 'wavelength') else '—'

                # Count replicates
                n_replicates = sum(len(f.get_kinetics_data()[1]) for f in decay_files)

                # Create item
                if isinstance(wavelength, (int, float)):
                    label = f"{compound_name} [{wavelength:.0f} nm, N={n_replicates}]"
                else:
                    label = f"{compound_name} [N={n_replicates}]"

                item = QTreeWidgetItem(self.decay_section, [label])
                item.setIcon(0, self.icon_decay)  # Add decay icon

                # Store data
                item.setData(0, Qt.ItemDataRole.UserRole, {
                    'type': 'decay_compound',
                    'compound': compound_name,
                    'files': decay_files
                })

        finally:
            self.tree.setUpdatesEnabled(True)

    def _populate_abs_section(self):
        """Populate absorption section."""
        self.tree.setUpdatesEnabled(False)

        try:
            self.abs_section.takeChildren()

            # Collect unique compounds with abs files
            abs_compounds = set()
            for compound_name, files in self.compounds.items():
                abs_files = [f for f in files if f.file_type == 'absorption']
                if abs_files:
                    abs_compounds.add(compound_name)

            for compound_name in sorted(abs_compounds):
                item = QTreeWidgetItem(self.abs_section, [compound_name])
                item.setIcon(0, self.icon_abs)  # Add absorption icon

                # Store data
                item.setData(0, Qt.ItemDataRole.UserRole, {
                    'type': 'abs_compound',
                    'compound': compound_name
                })

        finally:
            self.tree.setUpdatesEnabled(True)

    # ==================== RESULTS/PLOTS POPULATION ====================

    def add_result_item(self, result_type: str):
        """Add result item to results section."""
        # Check if already exists
        for i in range(self.results_section.childCount()):
            child = self.results_section.child(i)
            if child.text(0) == result_type:
                return  # Already exists

        # Create new item
        item = QTreeWidgetItem(self.results_section, [result_type])
        item.setIcon(0, self.icon_result)  # Add result icon
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'result',
            'result_type': result_type
        })

    def add_plot_item(self, plot_name: str, plot_data: dict):
        """Add plot item to plots section."""
        # Check if already exists
        for i in range(self.plots_section.childCount()):
            child = self.plots_section.child(i)
            if child.text(0) == plot_name:
                return  # Already exists

        # Create new item
        item = QTreeWidgetItem(self.plots_section, [plot_name])
        item.setIcon(0, self.icon_plot)  # Add plot icon
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'plot',
            'plot_name': plot_name,
            'plot_data': plot_data
        })

    # ==================== UTILITY METHODS ====================

    def get_checked_compounds(self) -> Dict[str, List[dict]]:
        """Get selected compounds from decay section (highlight-based)."""
        selected = {}

        # Get selected items from tree
        selected_items = self.tree.selectedItems()

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

        # Get selected items from tree
        selected_items = self.tree.selectedItems()

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

    def clear_tree(self):
        """Clear decay and abs files."""
        self.decay_section.takeChildren()
        self.abs_section.takeChildren()
        self.compounds = {}

    def clear_data(self):
        """Clear all loaded data."""
        self.clear_tree()
        self.current_folder = None

    def clear_results(self, clear_plots: bool = False):
        """Clear results and optionally plots."""
        self.results_section.takeChildren()

        if clear_plots:
            self.plots_section.takeChildren()

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
