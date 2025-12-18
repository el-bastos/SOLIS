#!/usr/bin/env python3
"""
SOLIS - Singlet Oxygen Luminescence Investigation System
Main GUI Application (PyQt6)
"""

import sys
import numpy as np
import json
from typing import List, Dict
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QStatusBar, QFileDialog, QMessageBox, QDockWidget,
    QToolBar, QToolButton, QStyle, QDialog, QProgressBar, QLabel, QTextBrowser,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, QSize, QTimer, QElapsedTimer
from pathlib import Path

from gui.file_browser_widget import FileBrowserWidget
from gui.results_viewer_widget import ResultsViewerWidget
from gui.preferences_dialog import PreferencesDialog
from gui.analysis_worker import AnalysisWorker
from gui.plot_viewer_widget import PlotViewerWidget
from gui.variable_study_widget import VariableStudyWidget
from gui.stylesheets import MENU_STYLE, PROGRESS_BAR_STYLE, get_global_flat_style
from utils.logger_config import get_logger
from utils.session_manager import SessionManager, save_session_dialog, load_session_dialog

logger = get_logger(__name__)


class SOLISMainWindow(QMainWindow):
    """Main application window for SOLIS."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOLIS - Singlet Oxygen Luminescence Investigation System")
        self.setGeometry(100, 100, 1200, 800)

        # Set application icon
        icon_path = Path(__file__).parent / 'logo' / 'SOLIS-ICON.png'
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Widgets
        self.file_browser = None
        self.results_viewer = None
        self.variable_study_widget = None
        self.analysis_worker = None

        # Plot viewers (floating windows)
        self.plot_viewers = {}  # {plot_id: PlotViewerWidget}

        # Plot operations log for session replay
        self.plot_operations = []  # List of plot operations to replay on session load

        # User preferences (default values)
        self.preferences = {
            'snr_thresholds': {
                'homogeneous': 5.0,
                'heterogeneous': 50.0
            },
            'surplus': {
                'mask_time_us': 6.0
            }
            # Heterogeneous analysis parameters now in HeterogeneousDialog (not preferences)
        }

        # Backward compatibility
        self.snr_thresholds = self.preferences['snr_thresholds']

        # Store loaded compounds for analysis
        self.loaded_compounds = None

        # Store analysis results for export
        self.analysis_results = None

        # Store mask corrections from preview plots
        self.mask_corrections = {}  # {compound_name or compound_repN: mask_end_time}

        # Flag for pending linearity check after analysis
        self._pending_linearity_check = False
        self._linearity_check_selection = []  # Store selected compounds for linearity check

        # Flag for pending surplus analysis after homogeneous
        self._pending_surplus_analysis = False

        # Setup UI components
        self._setup_menus()
        self._setup_toolbar()
        self._setup_status_bar()
        self._setup_widgets()

        # Initialize
        self.export_results_action.setEnabled(False)  # Disabled until analysis is complete
        self.status_label.setText("Ready")

    def _setup_menus(self):
        """Create menu bar with File, Analysis, Plots, Toolbars, and Preferences menus."""
        menubar = self.menuBar()

        # macOS-specific: Set native menu bar property
        # This ensures menus appear in the system menu bar on macOS
        if sys.platform == 'darwin':
            menubar.setNativeMenuBar(True)

        # Apply stylesheet for rectangular menus without shadows
        # Note: Shadow removal is platform-dependent and may not work on all systems
        menubar.setStyleSheet(MENU_STYLE)

        # === File Menu ===
        file_menu = menubar.addMenu("&File")

        # Open Folder
        open_folder_action = QAction("&Open Folder...", self)
        open_folder_action.setShortcut("Ctrl+O")
        open_folder_action.setStatusTip("Select folder containing data files")
        open_folder_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        # Save Session
        save_session_action = QAction("&Save Session...", self)
        save_session_action.setShortcut("Ctrl+S")
        save_session_action.setStatusTip("Save current session (data, results, settings) to file")
        save_session_action.triggered.connect(self._save_session)
        file_menu.addAction(save_session_action)
        self.save_session_action = save_session_action  # Store for enable/disable

        # Load Session
        load_session_action = QAction("&Load Session...", self)
        load_session_action.setShortcut("Ctrl+Shift+O")
        load_session_action.setStatusTip("Load previously saved session from file")
        load_session_action.triggered.connect(self._load_session)
        file_menu.addAction(load_session_action)

        file_menu.addSeparator()

        # Export Results
        export_results_action = QAction("&Export Results...", self)
        export_results_action.setShortcut("Ctrl+E")
        export_results_action.setStatusTip("Export kinetics results and statistics to CSV files")
        export_results_action.triggered.connect(self._export_results)
        file_menu.addAction(export_results_action)
        self.export_results_action = export_results_action  # Store for enable/disable

        file_menu.addSeparator()

        # Reset Session
        reset_action = QAction("&Reset Session", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.setStatusTip("Clear all data and start a fresh session")
        reset_action.triggered.connect(self._reset_session)
        file_menu.addAction(reset_action)

        file_menu.addSeparator()

        # Exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # === Analysis Menu ===
        analysis_menu = menubar.addMenu("&Analysis")

        # Homogeneous submenu
        homogeneous_menu = analysis_menu.addMenu("&Homogeneous")

        # Kinetics and QY
        kinetics_qy_action = QAction("&Kinetics and QY", self)
        kinetics_qy_action.setShortcut("Ctrl+H")
        kinetics_qy_action.setStatusTip("Run homogeneous kinetics analysis (SNR threshold: 5:1)")
        kinetics_qy_action.triggered.connect(lambda: self._run_analysis('homogeneous'))
        homogeneous_menu.addAction(kinetics_qy_action)

        # Linearity Check
        linearity_action = QAction("&Linearity Check", self)
        linearity_action.setShortcut("Ctrl+L")
        linearity_action.setStatusTip("Study variation of α with A(λ) or excitation intensity")
        linearity_action.triggered.connect(self._run_linearity_check)
        homogeneous_menu.addAction(linearity_action)

        # Heterogeneous (Vesicle) Analysis - direct action
        vesicle_action = QAction("He&terogeneous (Vesicle)", self)
        vesicle_action.setShortcut("Ctrl+V")
        vesicle_action.setStatusTip("Run heterogeneous vesicle analysis (3-exponential fit)")
        vesicle_action.triggered.connect(self._run_vesicle_analysis)
        analysis_menu.addAction(vesicle_action)

        # Surplus Analysis (for heterogeneous systems)
        analysis_menu.addSeparator()
        self.surplus_action = QAction("&Surplus Analysis", self)
        self.surplus_action.setShortcut("Ctrl+S")
        self.surplus_action.setStatusTip("Run surplus analysis on selected datasets for heterogeneous systems")
        self.surplus_action.setEnabled(True)  # Always enabled - runs homogeneous first if needed
        self.surplus_action.triggered.connect(self._run_surplus_analysis)
        analysis_menu.addAction(self.surplus_action)

        # === Preferences Menu ===
        preferences_menu = menubar.addMenu("&Preferences")

        # SNR Thresholds
        snr_thresholds_action = QAction("&SNR Thresholds...", self)
        snr_thresholds_action.setStatusTip("Configure SNR thresholds for analysis modes")
        snr_thresholds_action.triggered.connect(self._show_preferences)
        preferences_menu.addAction(snr_thresholds_action)

        # === Help Menu ===
        help_menu = menubar.addMenu("&Help")

        # Show Help
        show_help_action = QAction("&SOLIS Help", self)
        show_help_action.setShortcut("F1")
        show_help_action.setStatusTip("Show SOLIS help documentation")
        show_help_action.triggered.connect(self._show_help)
        help_menu.addAction(show_help_action)

    def _setup_toolbar(self):
        """Create toolbar with icon+text buttons for plotting and selection."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QSize(16, 16))  # Standard 16x16 icon size
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)  # Icon + text
        self.addToolBar(toolbar)

        # Load custom icons
        icon_dir = Path(__file__).parent / 'icons'

        # Select All Decays / Deselect All Decays button
        select_icon = QIcon(str(icon_dir / 'check_box_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg'))
        self.select_all_action = QAction(select_icon, "Select All Decays", self)
        self.select_all_action.setToolTip("Select/Deselect all decay compounds in browser")
        self.select_all_action.setEnabled(False)  # Disabled until data is loaded
        self.select_all_action.triggered.connect(self._toggle_select_all)
        toolbar.addAction(self.select_all_action)

        toolbar.addSeparator()

        # Individual Plots button (multiple separate plots)
        individual_icon = QIcon(str(icon_dir / 'lines_axis_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg'))
        self.toolbar_individual_plots_action = QAction(individual_icon, "Individual Plots", self)
        self.toolbar_individual_plots_action.setToolTip("Create separate plot windows for each selected item")
        self.toolbar_individual_plots_action.setEnabled(False)
        self.toolbar_individual_plots_action.triggered.connect(self._on_toolbar_individual_plots)
        toolbar.addAction(self.toolbar_individual_plots_action)

        # Plot Merged button (single merged plot)
        merged_icon = QIcon(str(icon_dir / 'line_axis_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg'))
        self.toolbar_plot_merged_action = QAction(merged_icon, "Plot Merged", self)
        self.toolbar_plot_merged_action.setToolTip("Combine all selected items into a single plot")
        self.toolbar_plot_merged_action.setEnabled(False)
        self.toolbar_plot_merged_action.triggered.connect(self._on_toolbar_plot_merged)
        toolbar.addAction(self.toolbar_plot_merged_action)

        # Store toolbar reference
        self.main_toolbar = toolbar

        # Track selection state for toggle
        self._all_mean_selected = False

    def _setup_status_bar(self):
        """Create enhanced status bar with progress indicator and timer."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create status label (left side - main status text)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)  # Stretch factor 1

        # Create progress bar (center - visual indicator with percentage inside)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(250)
        self.progress_bar.setMinimumWidth(200)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")  # Show percentage inside bar
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Hidden by default
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Create elapsed time label (right side - with separator)
        self.time_label = QLabel("")
        self.time_label.setMinimumWidth(80)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("margin-left: 10px;")
        self.time_label.setVisible(False)  # Hidden by default
        self.status_bar.addPermanentWidget(self.time_label)

        # Timer for elapsed time tracking
        self.elapsed_timer = QElapsedTimer()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_elapsed_time)

    def _update_elapsed_time(self):
        """Update elapsed time display."""
        if self.elapsed_timer.isValid():
            elapsed_ms = self.elapsed_timer.elapsed()
            elapsed_sec = elapsed_ms // 1000
            minutes = elapsed_sec // 60
            seconds = elapsed_sec % 60
            self.time_label.setText(f"⏱ {minutes:02d}:{seconds:02d}")

    def _start_progress(self):
        """Start progress tracking (show progress bar and timer)."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.time_label.setVisible(True)
        self.time_label.setText("⏱ 00:00")
        self.elapsed_timer.start()
        self.update_timer.start(500)  # Update every 500ms

    def _update_progress(self, percentage: int, status_text: str = ""):
        """Update progress bar and status text."""
        self.progress_bar.setValue(percentage)

        # Update status text if provided
        if status_text:
            self.status_label.setText(status_text)

    def _stop_progress(self, final_message: str = ""):
        """Stop progress tracking and hide progress widgets."""
        self.update_timer.stop()
        self.progress_bar.setVisible(False)
        self.time_label.setVisible(False)

        if final_message:
            self.status_label.setText(final_message)
        else:
            self.status_label.setText("Ready")

    def _set_busy_cursor(self):
        """Set waiting cursor to indicate app is busy (plot generation, etc.)."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _restore_cursor(self):
        """Restore normal cursor after busy operation completes."""
        QApplication.restoreOverrideCursor()

    def _setup_widgets(self):
        """Create and setup main widgets."""
        # Create split layout: File Browser (left) + Results Viewer (right)
        from PyQt6.QtWidgets import QSplitter

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: File Browser
        self.file_browser = FileBrowserWidget(parent=self)
        main_splitter.addWidget(self.file_browser)

        # RIGHT: Results Viewer
        self.results_viewer = ResultsViewerWidget(parent=self)
        main_splitter.addWidget(self.results_viewer)

        # Set splitter sizes (30% browser, 70% results)
        main_splitter.setSizes([300, 700])
        main_splitter.setStretchFactor(0, 0)  # Browser fixed-ish width
        main_splitter.setStretchFactor(1, 1)  # Results stretches

        # Wrap in QDockWidget for consistency
        self.data_browser_dock = QDockWidget("", self)  # No title text
        self.data_browser_dock.setWidget(main_splitter)
        self.data_browser_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )

        # Add to main window on the left side
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.data_browser_dock)

        # Connect file browser signals
        self.file_browser.status_message.connect(self._update_status)
        self.file_browser.preview_requested.connect(self._show_preview_plot)
        self.file_browser.folder_loaded.connect(self._on_folder_loaded)
        self.file_browser.absorption_plot_requested.connect(self._on_absorption_plot_requested)
        self.file_browser.absorption_merged_requested.connect(self._on_absorption_merged_requested)
        self.file_browser.result_item_clicked.connect(self._on_result_item_clicked)
        self.file_browser.plot_item_clicked.connect(self._on_plot_item_clicked)

        # Connect results viewer signals
        self.results_viewer.status_message.connect(self._update_status)
        self.results_viewer.plot_requested.connect(self._on_plot_requested)
        self.results_viewer.plot_merged_requested.connect(self._on_plot_merged_requested)
        self.results_viewer.surplus_plot_requested.connect(self._on_surplus_plot_requested)
        self.results_viewer.heterogeneous_plot_requested.connect(self._on_heterogeneous_plot_requested)

        # Start with browser hidden (opens after loading data)
        self.data_browser_dock.hide()

    def _on_result_item_clicked(self, result_type: str):
        """Handle result item click from file browser."""
        # Open result tab in results viewer
        self.results_viewer.open_result_tab(result_type)

    def _on_plot_item_clicked(self, plot_name: str, plot_data: dict):
        """Handle plot item click from file browser."""
        # Open plot tab in results viewer
        self.results_viewer.open_plot_tab(plot_name, plot_data)

    def _open_folder(self):
        """Open folder dialog and load data files."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if folder:
            self._load_data_folder(folder)

    def _load_data_folder(self, folder_path: str):
        """Load and parse data files from folder."""
        try:
            # Check if data is already loaded or analysis has been run
            if self.loaded_compounds is not None or self.results_viewer.kinetics_results:
                # Confirm with user that session will be reset
                reply = QMessageBox.question(
                    self,
                    "Reset Session",
                    "Opening a new folder will reset the current session.\n\n"
                    "All loaded data, analysis results, and plots will be cleared.\n\n"
                    "Do you want to continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )

                if reply == QMessageBox.StandardButton.No:
                    self.status_label.setText("Folder load cancelled")
                    return

                # User confirmed - reset session without additional confirmation
                logger.info("Resetting session before loading new folder")
                self._reset_session_internal()

            self.status_label.setText(f"Parsing files from: {folder_path}...")
            QApplication.processEvents()  # Update UI immediately

            # Load data through integrated browser (asynchronous - uses background thread)
            # Compounds will be available via folder_loaded signal
            self.file_browser.load_folder(folder_path)

        except Exception as e:
            error_msg = f"Error loading folder: {str(e)}"
            logger.error(error_msg)
            self.status_label.setText(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _on_folder_loaded(self, compounds: dict):
        """Handle folder loading completion (from background thread)."""
        try:
            # Store compounds for analysis
            self.loaded_compounds = compounds

            # Auto-open browser after successful loading
            self.data_browser_dock.show()

            # Enable Select All button (now data is loaded)
            self.select_all_action.setEnabled(True)

            logger.info(f"Folder loading completed with {len(compounds)} compounds")

        except Exception as e:
            error_msg = f"Error processing loaded folder: {str(e)}"
            logger.error(error_msg)
            self.status_label.setText(error_msg)

    def _update_status(self, message: str):
        """Update status bar with message from widgets."""
        self.status_label.setText(message)

    def _show_preferences(self):
        """Show preferences dialog."""
        dialog = PreferencesDialog(self, self.preferences)
        if dialog.exec():
            # User clicked OK - save new settings
            self.preferences = dialog.get_settings()
            self.snr_thresholds = self.preferences['snr_thresholds']  # Backward compatibility
            logger.info(f"Preferences updated: {self.preferences}")
            self.status_label.setText(
                f"Preferences updated: SNR thresholds and surplus mask time saved"
            )

    def _show_help(self):
        """Show help dialog with content from help.json."""
        dialog = HelpDialog(self)
        dialog.exec()

    def _export_results(self):
        """Export kinetics results and statistics to CSV files."""
        if not self.analysis_results:
            QMessageBox.warning(
                self,
                "No Results",
                "No analysis results available to export.\n\nPlease run an analysis first."
            )
            return

        # Ask user for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if not output_dir:
            return  # User cancelled

        try:
            from utils.csv_exporter import CSVExporter
            import os

            # Create exporter
            exporter = CSVExporter(output_dir=os.path.join(output_dir, "csv_exports"))

            # Get results from integrated browser
            kinetics_results = self.analysis_results['kinetics_results']
            statistics_results = self.analysis_results['statistics_results']
            qy_results = self.analysis_results['qy_results']

            # Prepare compounds_data dictionary for exporter
            compounds_data = {}
            for compound_name, compound_data in kinetics_results.items():
                # Get replicate results from 'results' key
                replicate_results = compound_data.get('results', [])

                # Get statistics data
                stats_data = statistics_results.get(compound_name, {})
                mean_arrays = stats_data.get('mean_arrays', {})
                sd_arrays = stats_data.get('sd_arrays', {})

                if replicate_results and mean_arrays and sd_arrays:
                    # Create a mock statistical analyzer object for csv_exporter
                    class MockStatisticalAnalyzer:
                        def __init__(self, mean_arrays, sd_arrays):
                            self._mean_arrays = mean_arrays
                            self._sd_arrays = sd_arrays

                        def get_mean_arrays(self):
                            return self._mean_arrays

                        def get_sd_arrays(self):
                            return self._sd_arrays

                    statistical_analyzer = MockStatisticalAnalyzer(mean_arrays, sd_arrays)

                    # Use compound name + wavelength as key
                    wavelength = compound_data.get('wavelength', 'unknown')
                    key = f"{compound_name}_{wavelength}nm"
                    compounds_data[key] = {
                        'replicate_results': replicate_results,
                        'statistical_analyzer': statistical_analyzer
                    }

            # Export all data
            logger.info(f"Exporting results to {output_dir}")
            self.status_label.setText("Exporting results...")

            # Prepare analysis_results for parameter statistics export
            # Convert to format expected by export_parameter_statistics:
            # {compound_name: [{'statistics': stats, 'wavelength': wl, 'classification': cls, 'mean_arrays': {...}}]}
            analysis_results_for_export = {}
            for compound_name in kinetics_results.keys():
                stats_data = statistics_results.get(compound_name, {})
                compound_data_from_kinetics = kinetics_results[compound_name]

                analysis_results_for_export[compound_name] = [{
                    'statistics': stats_data.get('statistics', {}),
                    'wavelength': compound_data_from_kinetics.get('wavelength', 'unknown'),
                    'classification': compound_data_from_kinetics.get('classification', 'unknown'),
                    'mean_arrays': stats_data.get('mean_arrays', {})
                }]

            # Export individual replicates and statistical summaries
            exported_files = exporter.export_all_compounds(
                compounds_data,
                analysis_results=analysis_results_for_export,
                qy_results=qy_results
            )

            # Count total files exported
            total_files = sum(len(v['individual_replicates']) for v in exported_files.values() if isinstance(v, dict))
            total_files += len([v for v in exported_files.values() if isinstance(v, str)])  # Add summary files

            # Show success message
            QMessageBox.information(
                self,
                "Export Complete",
                f"Results exported successfully!\n\n"
                f"Output directory: {output_dir}/csv_exports\n\n"
                f"Files exported:\n"
                f"  • Individual replicates: {total_files - 2} files\n"
                f"  • Parameter statistics: parameter_statistics.csv\n"
                f"  • Quantum yields: quantum_yields.csv"
            )

            logger.info(f"Export complete: {total_files} files")
            self.status_label.setText(f"Export complete: {total_files} CSV files saved")

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export results:\n\n{str(e)}"
            )
            self.status_label.setText("Export failed")

    def _save_session(self):
        """Save current session to JSON file."""
        # Check if there's anything to save
        if not self.loaded_compounds and not self.analysis_results:
            QMessageBox.information(
                self,
                "Nothing to Save",
                "No data or analysis results to save.\n\nPlease load data or run an analysis first."
            )
            return

        # Show save dialog
        filepath = save_session_dialog(parent=self)
        if not filepath:
            return  # User cancelled

        # Ask for optional description
        from PyQt6.QtWidgets import QInputDialog
        description, ok = QInputDialog.getText(
            self,
            "Session Description",
            "Enter an optional description for this session:",
            text=""
        )
        if not ok:
            description = ""

        try:
            # Get current data folder path
            folder_path = None
            if self.file_browser and hasattr(self.file_browser, 'current_folder'):
                folder_path = self.file_browser.current_folder

            # Extract heterogeneous results
            heterogeneous_results = None
            if self.file_browser and hasattr(self.results_viewer, 'heterogeneous_results'):
                heterogeneous_results = self.results_viewer.heterogeneous_results
                if heterogeneous_results:
                    logger.info(f"Saving {len(heterogeneous_results)} heterogeneous results")

            # Extract surplus results
            surplus_results = None
            if self.file_browser and hasattr(self.results_viewer, 'surplus_results'):
                surplus_results = self.results_viewer.surplus_results
                if surplus_results:
                    logger.info(f"Saving {len(surplus_results)} surplus results")

            # Extract plot window states
            plot_windows = self._get_plot_window_states()
            if plot_windows:
                logger.info(f"Saving {len(plot_windows)} plot window states")

            # Log plot operations for replay
            if self.plot_operations:
                logger.info(f"Saving {len(self.plot_operations)} plot operations for replay")

            # Save session
            success = SessionManager.save_session(
                filepath=filepath,
                loaded_compounds=self.loaded_compounds,
                analysis_results=self.analysis_results,
                heterogeneous_results=heterogeneous_results,
                surplus_results=surplus_results,
                preferences=self.preferences,
                mask_corrections=self.mask_corrections,
                folder_path=folder_path,
                description=description,
                plot_windows=plot_windows,
                plot_operations=self.plot_operations
            )

            if success:
                QMessageBox.information(
                    self,
                    "Session Saved",
                    f"Session saved successfully!\n\nFile: {filepath}"
                )
                logger.info(f"Session saved to {filepath}")
                self.status_label.setText(f"Session saved: {filepath.name}")
            else:
                QMessageBox.warning(
                    self,
                    "Save Failed",
                    "Failed to save session. Check logs for details."
                )

        except Exception as e:
            logger.error(f"Save session failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save session:\n\n{str(e)}"
            )

    def _load_session(self):
        """Load session from JSON file."""
        # Warn if current session has unsaved data
        if self.loaded_compounds or self.analysis_results:
            reply = QMessageBox.question(
                self,
                "Load Session",
                "Loading a session will replace current data.\n\n"
                "Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Show load dialog
        filepath = load_session_dialog(parent=self)
        if not filepath:
            return  # User cancelled

        try:
            # Load session
            session_data = SessionManager.load_session(filepath)
            if not session_data:
                QMessageBox.warning(
                    self,
                    "Load Failed",
                    "Failed to load session file. Check logs for details."
                )
                return

            # Reset current session first
            self._reset_session_internal()

            # Restore preferences
            if 'preferences' in session_data:
                self.preferences.update(session_data['preferences'])
                self.snr_thresholds = self.preferences['snr_thresholds']
                logger.info("Preferences restored from session")

            # Restore UI state
            if 'ui_state' in session_data:
                self.mask_corrections = session_data['ui_state'].get('mask_corrections', {})
                logger.info("UI state restored from session")

            # Restore loaded compounds (data)
            if session_data.get('data', {}).get('loaded_compounds'):
                self.loaded_compounds = session_data['data']['loaded_compounds']
                logger.info(f"Loaded compounds restored: {len(self.loaded_compounds)} compounds")

                # Update integrated browser with loaded data
                if self.file_browser:
                    # Show the data browser dock
                    self.data_browser_dock.show()
                    # Populate browser with loaded compounds
                    folder_path = session_data.get('data', {}).get('folder_path')
                    self.file_browser.populate_from_session(self.loaded_compounds, folder_path)
                    logger.info("Browser populated with loaded compounds from session")

            # Restore analysis results
            if session_data.get('analysis'):
                analysis_data = session_data['analysis']

                # Restore homogeneous analysis results
                if analysis_data.get('homogeneous'):
                    self.analysis_results = analysis_data['homogeneous']
                    logger.info("Homogeneous analysis results restored from session")
                    logger.info(f"Analysis results keys: {list(self.analysis_results.keys()) if isinstance(self.analysis_results, dict) else type(self.analysis_results)}")

                    if isinstance(self.analysis_results, dict):
                        if 'kinetics_results' in self.analysis_results:
                            logger.info(f"Found {len(self.analysis_results['kinetics_results'])} kinetics results")
                        if 'statistics_results' in self.analysis_results:
                            logger.info(f"Found {len(self.analysis_results['statistics_results'])} statistics results")
                        if 'qy_results' in self.analysis_results:
                            logger.info(f"Found {len(self.analysis_results['qy_results'])} QY results")

                    # Update integrated browser with homogeneous results
                    if self.file_browser and self.analysis_results:
                        self.results_viewer.populate_results_from_session(self.analysis_results)
                        logger.info("Results browser populated with homogeneous results from session")

                    # Enable toolbar buttons and actions if we have results
                    if self.analysis_results:
                        self.select_all_action.setEnabled(True)
                        self.toolbar_individual_plots_action.setEnabled(True)
                        self.toolbar_plot_merged_action.setEnabled(True)
                        self.export_results_action.setEnabled(True)
                        logger.info("Enabled toolbar actions for loaded analysis results")

                # Restore heterogeneous analysis results
                if analysis_data.get('heterogeneous') and self.file_browser:
                    heterogeneous_data = analysis_data['heterogeneous']
                    if not hasattr(self.results_viewer, 'heterogeneous_results'):
                        self.results_viewer.heterogeneous_results = {}

                    self.results_viewer.heterogeneous_results.update(heterogeneous_data)

                    # Update Tab 5 (Heterogeneous Results) in browser
                    if hasattr(self.results_viewer, 'populate_heterogeneous_results'):
                        self.results_viewer.populate_heterogeneous_results(
                            self.results_viewer.heterogeneous_results
                        )
                    logger.info(f"Restored {len(heterogeneous_data)} heterogeneous results")

                # Restore surplus analysis results
                if analysis_data.get('surplus') and self.file_browser:
                    surplus_data = analysis_data['surplus']
                    if not hasattr(self.results_viewer, 'surplus_results'):
                        self.results_viewer.surplus_results = {}

                    self.results_viewer.surplus_results.update(surplus_data)

                    # Update Surplus Results tab in browser
                    if hasattr(self.results_viewer, 'populate_surplus_results'):
                        self.results_viewer.populate_surplus_results(
                            self.results_viewer.surplus_results
                        )
                    logger.info(f"Restored {len(surplus_data)} surplus results")

            # Restore plot operations and replay them
            if 'ui_state' in session_data and 'plot_operations' in session_data['ui_state']:
                plot_operations = session_data['ui_state']['plot_operations']
                if plot_operations:
                    logger.info(f"Replaying {len(plot_operations)} plot operations")
                    self._replay_plot_operations(plot_operations)
            else:
                logger.debug("No plot operations to replay")

            # Show success message
            metadata = session_data.get('metadata', {})
            created = metadata.get('created', 'Unknown')
            desc = metadata.get('description', '')

            msg = f"Session loaded successfully!\n\nFile: {filepath}\nCreated: {created}"
            if desc:
                msg += f"\nDescription: {desc}"

            QMessageBox.information(self, "Session Loaded", msg)
            logger.info(f"Session loaded from {filepath}")
            self.status_label.setText(f"Session loaded: {filepath.name}")

        except Exception as e:
            logger.error(f"Load session failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load session:\n\n{str(e)}"
            )

    def _get_plot_window_states(self) -> List[Dict]:
        """
        Extract current state of all open plot windows WITH metadata for recreation.

        Returns:
            List of dictionaries with plot window states:
            {
                'plot_id': str,
                'title': str,
                'plot_type': str,  # 'individual', 'merged', 'mean', 'preview', etc.
                'plot_metadata': dict,  # Data needed to recreate plot
                'geometry': {'x': int, 'y': int, 'width': int, 'height': int},
                'visible': bool,
                'log_scale': bool
            }
        """
        plot_states = []

        for plot_id, viewer in self.plot_viewers.items():
            if viewer is None:
                continue

            state = {
                'plot_id': plot_id,
                'title': viewer.plot_title if hasattr(viewer, 'plot_title') else plot_id,
                'plot_type': getattr(viewer, 'plot_type', 'unknown'),
                'plot_metadata': getattr(viewer, 'plot_metadata', {}),
                'geometry': {
                    'x': viewer.x(),
                    'y': viewer.y(),
                    'width': viewer.width(),
                    'height': viewer.height()
                },
                'visible': viewer.isVisible(),
                'log_scale': viewer.current_log_x if hasattr(viewer, 'current_log_x') else True
            }

            plot_states.append(state)
            logger.debug(f"Saved state for plot '{plot_id}' (type: {state['plot_type']})")

        return plot_states

    def _restore_plot_window_states(self, plot_states: List[Dict]):
        """
        Recreate plots from saved metadata and restore window positions.

        Args:
            plot_states: List of plot window state dictionaries with recreation metadata

        Note:
            Attempts to recreate plots using saved metadata. If recreation fails,
            logs the plot information so user knows what was open before.
        """
        if not plot_states:
            return

        logger.info(f"Attempting to restore {len(plot_states)} plot windows")

        recreated_count = 0
        failed_plots = []

        for state in plot_states:
            plot_id = state.get('plot_id')
            plot_type = state.get('plot_type', 'unknown')
            plot_metadata = state.get('plot_metadata', {})

            logger.debug(f"Restoring plot '{plot_id}' (type: {plot_type})")

            # Try to recreate the plot based on type and metadata
            success = self._recreate_plot(plot_id, plot_type, plot_metadata, state)

            if success:
                recreated_count += 1
            else:
                failed_plots.append({
                    'plot_id': plot_id,
                    'plot_type': plot_type,
                    'title': state.get('title', plot_id)
                })

        logger.info(f"Plot restoration complete: {recreated_count}/{len(plot_states)} recreated")

        # If some plots couldn't be recreated, inform user
        if failed_plots:
            logger.warning(f"Could not auto-recreate {len(failed_plots)} plots:")
            for plot_info in failed_plots:
                logger.warning(f"  - {plot_info['title']} (type: {plot_info['plot_type']})")
            logger.info("These plots can be manually recreated from the results browser")

    def _recreate_plot(self, plot_id: str, plot_type: str, metadata: dict, state: dict) -> bool:
        """
        Attempt to recreate a single plot from saved metadata.

        Args:
            plot_id: Unique plot identifier
            plot_type: Type of plot ('individual', 'merged', 'mean', 'preview', etc.)
            metadata: Plot recreation data
            state: Full plot state dictionary (for geometry, visibility, etc.)

        Returns:
            True if plot was successfully recreated, False otherwise
        """
        try:
            # For now, we only attempt to recreate plots that are in the integrated browser
            # User can manually recreate complex plots
            #
            # Future enhancement: Implement recreation for each plot type
            # For Session 38, we'll log what was open and let user recreate manually

            logger.debug(f"Plot recreation not yet implemented for type '{plot_type}'")
            logger.debug(f"  Plot '{plot_id}' was open with data: {list(metadata.keys())}")

            return False

        except Exception as e:
            logger.error(f"Failed to recreate plot '{plot_id}': {e}")
            return False

    # ==================== Plot Operation Replay System ====================

    def _replay_plot_operations(self, operations: List[Dict]):
        """
        Replay saved plot operations to recreate plots.

        Args:
            operations: List of plot operation dictionaries

        Note:
            Each operation is replayed by calling the same analysis functions
            that created the original plot. This ensures plots are recreated
            identically to how they were originally generated.
        """
        if not operations:
            return

        logger.info(f"Replaying {len(operations)} plot operations")
        success_count = 0
        failed_count = 0

        for op in operations:
            op_type = op.get('type')
            params = op.get('params', {})

            try:
                if op_type == 'individual':
                    self._replay_individual_plot(params)
                elif op_type == 'mean':
                    self._replay_mean_plot(params)
                elif op_type == 'merged':
                    self._replay_merged_plot(params)
                elif op_type == 'surplus':
                    self._replay_surplus_plot(params)
                elif op_type == 'heterogeneous_fit':
                    self._replay_heterogeneous_fit_plot(params)
                elif op_type == 'heterogeneous_landscape':
                    self._replay_heterogeneous_landscape_plot(params)
                elif op_type == 'linearity':
                    self._replay_linearity_plot(params)
                elif op_type == 'absorption_single':
                    self._replay_absorption_single_plot(params)
                elif op_type == 'absorption_merged':
                    self._replay_absorption_merged_plot(params)
                else:
                    logger.warning(f"Unknown plot operation type: {op_type}")
                    failed_count += 1
                    continue

                success_count += 1

            except Exception as e:
                logger.error(f"Failed to replay {op_type} plot: {e}")
                failed_count += 1

        logger.info(f"Plot replay complete: {success_count}/{len(operations)} succeeded, {failed_count} failed")

    def _replay_individual_plot(self, params: dict):
        """Replay individual replicate plot."""
        compound = params['compound']
        replicate_num = params['replicate_num']

        # Look up result from session data
        kinetics_results = self.results_viewer.kinetics_results
        if compound not in kinetics_results:
            logger.warning(f"Cannot replay plot: compound '{compound}' not found")
            return

        replicates = kinetics_results[compound]
        if replicate_num >= len(replicates):
            logger.warning(f"Cannot replay plot: replicate {replicate_num} not found for '{compound}'")
            return

        result = replicates[replicate_num]

        # Create plot using existing plotter
        from plotting.solis_plotter import SOLISPlotter
        plotter = SOLISPlotter()
        fig = plotter.plot_single_decay_mpl(
            result,
            log_x=False,
            title=f"{compound} - Rep {replicate_num}"
        )

        # Display (don't re-log this operation!)
        self._show_plot(
            f"{compound}_Rep{replicate_num}",
            f"{compound} Rep {replicate_num}",
            fig,
            skip_logging=True
        )

    def _replay_mean_plot(self, params: dict):
        """Replay mean plot."""
        compound = params['compound']

        # Look up statistics from session data
        statistics_results = self.results_viewer.statistics_results
        if compound not in statistics_results:
            logger.warning(f"Cannot replay plot: compound '{compound}' statistics not found")
            return

        stats_data = statistics_results[compound]
        mean_arrays = stats_data['mean_arrays']
        sd_arrays = stats_data['sd_arrays']

        # Create plot using existing plotter
        from plotting.solis_plotter import SOLISPlotter
        plotter = SOLISPlotter()
        fig = plotter.plot_mean_decay_mpl(
            mean_arrays,
            sd_arrays,
            log_x=False,
            title=f"{compound} - Mean ± SD"
        )

        # Display
        self._show_plot(
            f"{compound}_Mean",
            f"Mean: {compound}",
            fig,
            skip_logging=True
        )

    def _replay_merged_plot(self, params: dict):
        """Replay merged plot."""
        items_saved = params.get('items', [])
        if not items_saved:
            logger.warning("Cannot replay merged plot: no items in params")
            return

        # Reconstruct selected_items with full data
        selected_items = []
        for saved_item in items_saved:
            compound = saved_item['compound']
            item_type = saved_item.get('type', 'replicate')
            replicate_num = saved_item.get('replicate_num')

            if item_type == 'mean':
                # Get mean data
                statistics_results = self.results_viewer.statistics_results
                if compound not in statistics_results:
                    logger.warning(f"Cannot find mean data for '{compound}'")
                    continue

                stats_data = statistics_results[compound]
                selected_items.append({
                    'type': 'mean',
                    'compound': compound,
                    'mean_arrays': stats_data['mean_arrays'],
                    'sd_arrays': stats_data['sd_arrays']
                })

            else:  # replicate
                # Get replicate data
                kinetics_results = self.results_viewer.kinetics_results
                if compound not in kinetics_results:
                    logger.warning(f"Cannot find kinetics results for '{compound}'")
                    continue

                replicates = kinetics_results[compound]
                if replicate_num is None or replicate_num >= len(replicates):
                    logger.warning(f"Invalid replicate number for '{compound}'")
                    continue

                result = replicates[replicate_num]
                selected_items.append({
                    'type': 'replicate',
                    'compound': compound,
                    'replicate_num': replicate_num,
                    'result': result
                })

        if not selected_items:
            logger.warning("Cannot replay merged plot: no valid items found")
            return

        # Create merged plot
        from plotting.solis_plotter import SOLISPlotter
        plotter = SOLISPlotter()
        fig = plotter.plot_merged_decay_mpl(
            selected_items,
            log_x=True,
            title="Merged Decay Curves"
        )

        # Generate plot_id
        compounds = [item['compound'] for item in selected_items]
        if len(compounds) > 3:
            plot_id = f"Merged_{len(compounds)}_datasets"
            title = f"Merged Plot ({len(compounds)} datasets)"
        else:
            plot_id = "Merged_" + "_".join(compounds)
            title = "Merged: " + ", ".join(compounds)

        self._show_plot(plot_id, title, fig, skip_logging=True)

    def _replay_surplus_plot(self, params: dict):
        """Replay surplus analysis plot."""
        compound = params['compound']

        # Get surplus result
        surplus_results = self.results_viewer.surplus_results
        if not surplus_results or compound not in surplus_results:
            logger.warning(f"Cannot replay surplus plot: no result for '{compound}'")
            return

        result = surplus_results[compound]

        # Create plot
        from plotting.solis_plotter import SOLISPlotter
        plotter = SOLISPlotter()
        fig = plotter.plot_surplus_analysis_mpl(result, log_x=True)

        # Show plot
        plot_id = f"surplus_{compound}"
        title = f"Surplus Analysis - {compound}"
        self._show_plot(plot_id, title, fig, skip_logging=True)

    def _replay_heterogeneous_fit_plot(self, params: dict):
        """Replay heterogeneous fit plot (uses saved result, NOT re-running grid search)."""
        key = params['key']

        # Get heterogeneous result (already computed and saved in session)
        heterogeneous_results = self.results_viewer.heterogeneous_results
        if not heterogeneous_results or key not in heterogeneous_results:
            logger.warning(f"Cannot replay heterogeneous fit: no result for '{key}'")
            return

        result = heterogeneous_results[key]

        # Create plotter and generate fit plot
        from heterogeneous.heterogeneous_plotter_new import HeterogeneousPlotter
        plotter = HeterogeneousPlotter(result)
        fig = plotter.plot_fit_curves(show_components=True)

        # Show plot
        plot_id = f"hetero_fit_{key}"
        title = f"Heterogeneous Fit - {key}"
        self._show_plot(plot_id, title, fig, skip_logging=True)

    def _replay_heterogeneous_landscape_plot(self, params: dict):
        """Replay heterogeneous landscape plot (uses saved result)."""
        key = params['key']

        # Get heterogeneous result
        heterogeneous_results = self.results_viewer.heterogeneous_results
        if not heterogeneous_results or key not in heterogeneous_results:
            logger.warning(f"Cannot replay heterogeneous landscape: no result for '{key}'")
            return

        result = heterogeneous_results[key]

        # Create plotter and generate landscape plot
        from heterogeneous.heterogeneous_plotter_new import HeterogeneousPlotter
        plotter = HeterogeneousPlotter(result)

        try:
            fig = plotter.plot_figure_4()
            plot_id = f"hetero_grid_{key}"
            title = f"Chi-square Landscape - {key}"
            self._show_plot(plot_id, title, fig, skip_logging=True)
        except ValueError as e:
            logger.warning(f"Could not replay chi-square landscape: {e}")

    def _replay_linearity_plot(self, params: dict):
        """Replay linearity plot."""
        check_type = params['check_type']
        compounds = params['compounds']

        # Reconstruct selected_items (need full data with statistics)
        from plotting.variable_study_plotter import plot_alpha_vs_absorption_mpl, plot_alpha_vs_intensity_mpl

        selected_items = []
        kinetics_results = self.results_viewer.kinetics_results
        statistics_results = self.results_viewer.statistics_results

        ei_unit = None

        for compound_name in compounds:
            if compound_name not in kinetics_results or compound_name not in statistics_results:
                logger.warning(f"Cannot replay linearity: data missing for '{compound_name}'")
                continue

            replicates = kinetics_results[compound_name]
            stats_data = statistics_results[compound_name]

            # Get A(λ) and EI from first replicate (representative)
            if not replicates:
                continue

            first_replicate = replicates[0]
            decay_file = first_replicate.parsed_file if hasattr(first_replicate, 'parsed_file') else None

            if not decay_file:
                continue

            a_lambda = decay_file.absorption_at_wavelength if decay_file.absorption_at_wavelength is not None else 0
            ei_value = decay_file.excitation_intensity if decay_file.excitation_intensity is not None else None

            if ei_unit is None and decay_file.intensity_unit:
                ei_unit = decay_file.intensity_unit

            ei_str = ""
            if ei_value is not None and ei_value > 0:
                ei_str = f"{ei_value}"
                if ei_unit:
                    ei_str += f" {ei_unit}"

            # Build item
            item = {
                'type': 'mean',
                'compound': compound_name,
                'wavelength': 0,
                'classification': '',
                'mean_arrays': stats_data['mean_arrays'],
                'sd_arrays': stats_data['sd_arrays'],
                'abs_at_wavelength': f"{a_lambda:.3f}",
                'excitation_intensity': ei_str,
                'alpha_mean': stats_data['statistics'].get('A_mean', 0),
                'alpha_sd': stats_data['statistics'].get('A_sd', 0),
                'analysis_result': stats_data
            }
            selected_items.append(item)

        if not selected_items:
            logger.warning("Cannot replay linearity plot: no valid data")
            return

        # Create appropriate plot
        try:
            if check_type == 'beer_lambert':
                fig, _ = plot_alpha_vs_absorption_mpl(selected_items, ei_unit or "")
                plot_id = "Alpha_vs_Absorption"
                title = "α vs (1 - 10^(-A(λ)))"
            elif check_type == 'excitation_intensity':
                fig, _ = plot_alpha_vs_intensity_mpl(selected_items, ei_unit or "")
                plot_id = "Alpha_vs_Intensity"
                title = "α vs Excitation Intensity"
            else:
                logger.warning(f"Unknown linearity check type: {check_type}")
                return

            self._show_plot(plot_id, title, fig, skip_logging=True)

        except Exception as e:
            logger.error(f"Failed to replay linearity plot: {e}")

    def _replay_absorption_single_plot(self, params: dict):
        """Replay single absorption spectrum plot."""
        compound = params['compound']
        excitation_wavelengths = params.get('excitation_wavelengths', [])

        # Look up absorption file from loaded_compounds
        if compound not in self.loaded_compounds:
            logger.warning(f"Cannot replay absorption plot: compound '{compound}' not found")
            return

        abs_files = [f for f in self.loaded_compounds[compound] if f.file_type == 'absorption']
        if not abs_files:
            logger.warning(f"Cannot replay absorption plot: no absorption file for '{compound}'")
            return

        abs_file = abs_files[0]

        # Create plot
        from plotting.solis_plotter import SOLISPlotter
        plotter = SOLISPlotter()

        try:
            fig = plotter.plot_absorption_spectrum_mpl(
                abs_file,
                excitation_wavelengths=excitation_wavelengths
            )

            plot_id = f"abs_{compound}"
            title = f"Absorption Spectrum - {compound}"
            self._show_plot(plot_id, title, fig, skip_logging=True)

        except Exception as e:
            logger.error(f"Failed to replay absorption spectrum plot: {e}")

    def _replay_absorption_merged_plot(self, params: dict):
        """Replay merged absorption spectra plot."""
        compounds = params['compounds']
        excitation_wavelengths = params.get('excitation_wavelengths', {})

        # Look up absorption files for all compounds
        parsed_files = []
        for compound in compounds:
            if compound not in self.loaded_compounds:
                logger.warning(f"Skipping compound '{compound}': not found in loaded data")
                continue

            abs_files = [f for f in self.loaded_compounds[compound] if f.file_type == 'absorption']
            if abs_files:
                parsed_files.append(abs_files[0])
            else:
                logger.warning(f"Skipping compound '{compound}': no absorption file")

        if not parsed_files:
            logger.warning("Cannot replay merged absorption plot: no absorption files found")
            return

        # Create plot
        from plotting.solis_plotter import SOLISPlotter
        plotter = SOLISPlotter()

        try:
            fig = plotter.plot_merged_absorption_spectra_mpl(
                parsed_files,
                excitation_wavelengths=excitation_wavelengths
            )

            plot_id = "abs_merged"
            title = f"Merged Absorption Spectra ({len(parsed_files)} compounds)"
            self._show_plot(plot_id, title, fig, skip_logging=True)

        except Exception as e:
            logger.error(f"Failed to replay merged absorption spectra plot: {e}")

    # ==================== End of Plot Replay System ====================

    def _reset_session(self):
        """Clear all data and reset the application to initial state (with confirmation)."""
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Reset Session",
            "Are you sure you want to reset the session?\n\n"
            "This will clear all loaded data, analysis results, and close all plot windows.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return

        self._reset_session_internal()

    def _reset_session_internal(self):
        """Internal method to reset session without confirmation dialog."""
        logger.info("Resetting session")

        # Stop any running analysis
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.stop()
            self.analysis_worker.wait()
            self.analysis_worker = None

        # Destroy all plot data and close viewers
        for viewer in self.plot_viewers.values():
            viewer.destroy_plot()  # Clear all data and browser cache
            viewer.close()
        self.plot_viewers.clear()

        # Close and clear variable study widget
        if self.variable_study_widget:
            self.variable_study_widget.close()
            self.variable_study_widget = None

        # Clear integrated browser (all sections including plots)
        if self.file_browser:
            self.file_browser.clear_data()
            self.file_browser.clear_results(clear_plots=True); self.results_viewer.clear_results(clear_plots=True); self.results_viewer.clear_results(clear_plots=True)  # Clear plots for full reset
            self.data_browser_dock.hide()

        # Clear loaded data
        self.loaded_compounds = None
        self.analysis_results = None
        self.mask_corrections.clear()

        # Clear plot operations log
        self.plot_operations.clear()

        # Disable and reset toolbar buttons
        self.select_all_action.setEnabled(False)
        self.export_results_action.setEnabled(False)
        # Reset to select icon
        style = self.style()
        select_icon = style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        self.select_all_action.setIcon(select_icon)
        self.select_all_action.setToolTip("Select/Deselect all compounds in current tab")
        self.toolbar_individual_plots_action.setEnabled(False)
        self.toolbar_plot_merged_action.setEnabled(False)
        self._all_mean_selected = False

        # Reset status bar
        self.status_label.setText("Session reset - Ready for new data")
        logger.info("Session reset complete")

    def _run_analysis(self, mode: str):
        """
        Run analysis for selected compounds.

        Parameters
        ----------
        mode : str
            'homogeneous' or 'heterogeneous'
        """
        # Check if data is loaded
        if not self.loaded_compounds:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load data files first using File → Open Folder."
            )
            return

        # Get selected compounds from integrated browser (same wavelength only)
        selected_replicates = self.file_browser.get_checked_compounds()

        if not selected_replicates:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select compounds to analyze by checking them in the Data Browser."
            )
            return

        # Get SNR threshold for this mode
        snr_threshold = self.snr_thresholds[mode]

        logger.info(f"Starting {mode} analysis (SNR threshold: {snr_threshold}:1)")

        # Start progress tracking
        self._start_progress()
        self.status_label.setText(f"Running {mode} analysis...")

        # Clear previous results in integrated browser
        self.file_browser.clear_results()

        # Create and start analysis worker (pass mask corrections)
        self.analysis_worker = AnalysisWorker(
            selected_replicates,
            snr_threshold,
            mode,
            mask_corrections=self.mask_corrections
        )

        # Connect worker signals
        self.analysis_worker.progress_update.connect(self._on_progress_update)
        self.analysis_worker.snr_calculated.connect(self._on_snr_calculated)
        self.analysis_worker.replicate_analyzed.connect(self._on_replicate_analyzed)
        self.analysis_worker.analysis_complete.connect(self._on_analysis_complete)
        self.analysis_worker.error_occurred.connect(self._on_analysis_error)

        # Start worker thread
        self.analysis_worker.start()

        # Ensure browser is visible to show results later
        if not self.data_browser_dock.isVisible():
            self.data_browser_dock.show()

        # Keep main window in focus
        self.raise_()
        self.activateWindow()

    def _on_progress_update(self, percentage: int, status: str):
        """Handle progress updates from analysis worker."""
        self._update_progress(percentage, status)

    def _on_snr_calculated(self, compound_name: str, replicate_num: int, snr_linear: float):
        """Handle SNR calculation - log only (SNR will be shown in Kinetics Results tab after analysis)."""
        logger.debug(f"SNR calculated for {compound_name} Rep{replicate_num}: {snr_linear:.1f}:1")

    def _on_replicate_analyzed(self, compound_name: str, replicate_num: int, result_dict: dict):
        """Handle individual replicate analysis completion."""
        # Could add live logging here if needed
        pass

    def _on_analysis_complete(self, results: dict):
        """Handle analysis completion - populate kinetics and QY tabs in integrated browser."""
        logger.info("Analysis complete - populating results")

        # Store results for export
        self.analysis_results = results

        # Extract results
        kinetics_results = results['kinetics_results']
        statistics_results = results['statistics_results']
        qy_results = results['qy_results']
        excluded_count = results['excluded_count']

        # Populate results viewer (results section and QY)
        self.results_viewer.populate_kinetics_results(kinetics_results, statistics_results)
        self.results_viewer.populate_qy_results(qy_results)

        # Add result items to file browser
        self.file_browser.add_result_item('Kinetics')
        self.file_browser.add_result_item('Quantum Yield')

        # Auto-open Kinetics tab to show results
        self.results_viewer.open_result_tab('Kinetics')

        # Enable toolbar buttons and export action
        self.select_all_action.setEnabled(True)
        self.toolbar_individual_plots_action.setEnabled(True)
        self.toolbar_plot_merged_action.setEnabled(True)
        self.export_results_action.setEnabled(True)  # Enable CSV export

        # Show completion message
        total_compounds = len(kinetics_results)
        total_qy = len(qy_results)

        QMessageBox.information(
            self,
            "Analysis Complete",
            f"Analysis completed successfully!\n\n"
            f"Compounds analyzed: {total_compounds}\n"
            f"Quantum yields calculated: {total_qy}\n"
            f"Replicates excluded (SNR < {results['snr_threshold']}:1): {excluded_count}\n\n"
            f"Tip: Check boxes in Kinetics Results tab to view plots."
        )

        # Stop progress tracking
        self._stop_progress("Analysis complete!")

        # Check if linearity check was pending
        if self._pending_linearity_check:
            self._pending_linearity_check = False
            # Run linearity check now that analysis is complete
            self._perform_automatic_linearity_check()

        # Check if surplus analysis was pending
        if self._pending_surplus_analysis:
            self._pending_surplus_analysis = False
            logger.info("Homogeneous complete - auto-running surplus analysis now")
            # Run surplus analysis now that homogeneous is complete
            self._execute_surplus_analysis()

    def _on_view_plot_from_kinetics(self):
        """Handle Individual Plots button click - get selection from results_viewer."""
        selected_items = self.results_viewer.get_selected_items_for_plotting()
        if selected_items:
            self._on_plot_requested(selected_items)
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Selection",
                              "Please select at least one item from the Kinetics tab to plot.")

    def _on_plot_merged_from_kinetics(self):
        """Handle Plot Merged button click - get selection from results_viewer."""
        selected_items = self.results_viewer.get_selected_items_for_plotting()
        if selected_items:
            self._on_plot_merged_requested(selected_items)
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Selection",
                              "Please select at least one item from the Kinetics tab to plot.")

    def _on_plot_requested(self, selected_items: list):
        """
        Handle plot request from Results Widget.
        Create ONE plot with all selected items (mean and/or individual replicates).
        """
        from plotting.solis_plotter import SOLISPlotter

        if not selected_items:
            return

        # Show progress feedback and busy cursor
        total_plots = len(selected_items)
        self.status_label.setText(f"Generating {total_plots} plot(s)...")
        self._set_busy_cursor()
        QApplication.processEvents()  # Update UI immediately

        plotter = SOLISPlotter()

        # For now: simple implementation - one plot per selection type
        # TODO: Support multiple curves in one plot
        for idx, item in enumerate(selected_items, start=1):
            # Update status for each plot
            self.status_label.setText(f"Generating plot {idx}/{total_plots}...")
            QApplication.processEvents()
            if item['type'] == 'mean':
                # Log operation for replay
                operation = {
                    'type': 'mean',
                    'params': {'compound': item['compound']}
                }
                self.plot_operations.append(operation)

                # Plot MEAN curves with SD envelopes
                fig = plotter.plot_mean_decay_mpl(
                    item['mean_arrays'],
                    item['sd_arrays'],
                    log_x=False,
                    title=f"{item['compound']} - Mean ± SD"
                )
                self._show_plot(f"{item['compound']}_Mean", f"Mean: {item['compound']}", fig)

            elif item['type'] == 'replicate':
                # Log operation for replay
                operation = {
                    'type': 'individual',
                    'params': {
                        'compound': item['compound'],
                        'replicate_num': item['replicate_num']
                    }
                }
                self.plot_operations.append(operation)

                # Plot individual replicate
                fig = plotter.plot_single_decay_mpl(
                    item['result'],
                    log_x=False,
                    title=f"{item['compound']} - Rep {item['replicate_num']}"
                )
                self._show_plot(
                    f"{item['compound']}_Rep{item['replicate_num']}",
                    f"{item['compound']} Rep {item['replicate_num']}",
                    fig
                )

        # Done - restore cursor
        self._restore_cursor()
        self.status_label.setText(f"{total_plots} plot(s) generated")

    def _on_plot_merged_requested(self, selected_items: list):
        """
        Handle merged plot request from Results Widget.
        Create a SINGLE plot with all selected datasets and stacked residual panels.
        """
        from plotting.solis_plotter import SOLISPlotter

        if not selected_items:
            return

        # Show progress feedback and busy cursor
        n_datasets = len(selected_items)
        self.status_label.setText(f"Generating merged plot ({n_datasets} datasets)...")
        self._set_busy_cursor()
        QApplication.processEvents()  # Update UI immediately

        plotter = SOLISPlotter()

        # Create merged plot with all selected items
        try:
            # Log operation for replay
            items_for_log = []
            for item in selected_items:
                items_for_log.append({
                    'compound': item['compound'],
                    'replicate_num': item.get('replicate_num'),
                    'type': item.get('type')  # 'replicate' or 'mean'
                })
            operation = {
                'type': 'merged',
                'params': {'items': items_for_log}
            }
            self.plot_operations.append(operation)

            # Use matplotlib version (Session 15)
            fig = plotter.plot_merged_decay_mpl(
                selected_items,
                log_x=True,
                title="Merged Decay Curves"
            )

            # Generate plot_id from all compounds
            compounds = [item['compound'] for item in selected_items]
            if len(compounds) > 3:
                plot_id = f"Merged_{len(compounds)}_datasets"
                title = f"Merged Plot ({len(compounds)} datasets)"
            else:
                plot_id = "Merged_" + "_".join(compounds)
                title = "Merged: " + ", ".join(compounds)

            self._show_plot(plot_id, title, fig)

            # Done - restore cursor
            self._restore_cursor()
            self.status_label.setText(f"Merged plot generated ({n_datasets} datasets)")

        except Exception as e:
            self._restore_cursor()  # Restore cursor on error too
            self.status_label.setText("Merged plot failed")
            QMessageBox.critical(
                self,
                "Plot Error",
                f"Failed to create merged plot:\n{str(e)}"
            )
            logger.exception("Error creating merged plot")

    def _show_plot(self, plot_id: str, title: str, fig, skip_logging: bool = False):
        """
        Show a plot in a viewer window and add to browser.

        Args:
            plot_id: Unique identifier for the plot
            title: Window title
            fig: Matplotlib figure
            skip_logging: If True, don't log this operation (for replay operations)
        """
        # Add plot to integrated browser
        if self.file_browser:
            plot_data = {
                'figure': fig,
                'name': plot_id,
                'title': title
            }
            # Format plot name based on type:
            # Kinetics_TMPyP_Mean, Surplus_TMPyP, Heterogeneous_TMPyP_Rep1, Absorption_TMPyP
            if plot_id.startswith('surplus_'):
                # surplus_TMPyP → Surplus_TMPyP
                compound = plot_id.replace('surplus_', '')
                plot_name = f"Surplus_{compound}"
            elif plot_id.startswith('hetero_fit_'):
                # hetero_fit_TMPyP_Rep1 → Heterogeneous_TMPyP_Rep1
                key = plot_id.replace('hetero_fit_', '')
                plot_name = f"Heterogeneous_{key}"
            elif plot_id.startswith('hetero_grid_'):
                # hetero_grid_TMPyP_Rep1 → Heterogeneous_Landscape_TMPyP_Rep1
                key = plot_id.replace('hetero_grid_', '')
                plot_name = f"Heterogeneous_Landscape_{key}"
            elif plot_id.startswith('abs_merged_'):
                # abs_merged_TMPyP_PN → Absorption_Merged_TMPyP_PN
                # abs_merged_3 → Absorption_Merged_All
                if plot_id.split('_')[-1].isdigit():
                    plot_name = "Absorption_Merged_All"
                else:
                    plot_name = f"Absorption_{plot_id.replace('abs_', '')}"
            elif plot_id.startswith('abs_'):
                # abs_TMPyP → Absorption_TMPyP
                compound = plot_id.replace('abs_', '')
                plot_name = f"Absorption_{compound}"
            elif plot_id.startswith('Merged_'):
                # Merged_TMPyP_Compound2 → Kinetics_Merged_TMPyP_Compound2
                # Merged_3_datasets → Kinetics_Merged_All
                if '_datasets' in plot_id:
                    plot_name = "Kinetics_Merged_All"
                else:
                    plot_name = f"Kinetics_{plot_id}"
            else:
                # Default: Kinetics_TMPyP_Mean, Kinetics_TMPyP_Rep1
                plot_name = f"Kinetics_{plot_id}"

            self.file_browser.add_plot_item(plot_name, plot_data)

            # Open plot in browser tab
            self.results_viewer.open_plot_tab(plot_name, plot_data)

        self.status_label.setText(f"Plot displayed: {plot_name}")

    def _on_analysis_error(self, error_message: str):
        """Handle analysis error."""
        logger.error(f"Analysis error: {error_message}")

        # Stop progress tracking
        self._stop_progress(f"Analysis error: {error_message}")

        QMessageBox.critical(
            self,
            "Analysis Error",
            f"An error occurred during analysis:\n\n{error_message}"
        )

    def _show_mean_plot(self, compound_name: str, replicate_results: list):
        """
        Show mean curve plot with SD envelope for a compound.

        Parameters
        ----------
        compound_name : str
            Name of the compound
        replicate_results : list
            List of KineticsResult objects for all replicates
        """
        try:
            from plotting.solis_plotter import SOLISPlotter

            plot_id = f"{compound_name}_Mean"

            # Create or reuse plot viewer
            if plot_id not in self.plot_viewers:
                viewer = PlotViewerWidget(self, plot_title=f"Mean: {compound_name}")
                viewer.window_closed.connect(lambda: self._on_plot_closed(plot_id))
                self.plot_viewers[plot_id] = viewer

            viewer = self.plot_viewers[plot_id]

            # Create batch plot with mean and SD using matplotlib (Session 15)
            plotter = SOLISPlotter()
            logger.info(f"Creating batch plot for {compound_name} with {len(replicate_results)} replicates")
            fig = plotter.plot_batch_summary_mpl(
                replicate_results,
                log_x=False,
                title=f'{compound_name} - Mean ± SD (n={len(replicate_results)})',
                show_mean=True,
                show_statistics=True
            )

            logger.info(f"Batch plot created successfully, figure type: {type(fig)}")

            # Update plot data for future re-plotting
            viewer.current_data = {
                'file_type': 'decay',
                'preview_mode': False,
                'results': replicate_results,  # Store all results for batch plot
                'compound_name': compound_name
            }

            # Display the figure
            logger.info(f"Calling show_matplotlib_figure...")
            viewer.show_matplotlib_figure(fig)
            logger.info(f"show_matplotlib_figure completed")

            viewer.show()
            viewer.raise_()
            viewer.activateWindow()

            self.status_label.setText(f"Mean plot opened: {compound_name}")
            logger.info(f"Mean plot window shown and activated")

        except Exception as e:
            logger.error(f"Failed to create mean plot: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Plot Error",
                f"Failed to create mean plot:\n\n{str(e)}"
            )

    def _run_linearity_check(self):
        """
        Automatic linearity check with selection-first workflow.

        Workflow:
        1. User selects datasets in browser (checkboxes)
        2. Run kinetics analysis automatically if not done yet
        3. Detect which parameter varies (A or EI) using CV > 10%
        4. Generate appropriate plots automatically
        """
        # Check if data is loaded
        if not self.loaded_compounds:
            QMessageBox.warning(
                self,
                "No Data Loaded",
                "Please load data from a folder first (File → Open Folder)."
            )
            return

        # Get selected compounds from browser (Tab 1 - Loaded Files)
        selected_replicates = self.file_browser.get_checked_compounds()

        if not selected_replicates:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select compounds in the Browser (Tab 1: Loaded Files) "
                "by checking the boxes next to the compound names.\n\n"
                "These will be used for linearity analysis."
            )
            return

        # Store selected compounds for linearity check
        self._linearity_check_selection = list(selected_replicates.keys())
        logger.info(f"Linearity check: selected {len(self._linearity_check_selection)} compounds: {self._linearity_check_selection}")

        # Check if analysis has been run - if not, run it automatically (no prompt)
        if not self.results_viewer.kinetics_results:
            logger.info("Linearity check: running kinetics analysis automatically...")
            self.status_label.setText("Running kinetics analysis for linearity check...")

            # Run analysis in homogeneous mode automatically
            self._pending_linearity_check = True
            self._run_analysis('homogeneous')
            return

        # Perform automatic linearity check with selected compounds
        self._perform_automatic_linearity_check()

    def _perform_automatic_linearity_check(self):
        """
        Perform automatic linearity check by detecting which parameter varies.

        Algorithm:
        1. Collect SELECTED datasets with their A(λ) and EI values
        2. Calculate coefficient of variation (CV) for both A and EI
        3. Determine which varies more (CV > 10% threshold)
        4. Generate appropriate plots automatically
        """
        from plotting.variable_study_plotter import plot_alpha_vs_absorption_mpl, plot_alpha_vs_intensity_mpl

        logger.info("Starting automatic linearity check")
        self.status_label.setText("Analyzing parameter variation...")

        # Collect datasets from kinetics results
        kinetics_results = self.results_viewer.kinetics_results
        statistics_results = self.results_viewer.statistics_results

        if not kinetics_results:
            QMessageBox.warning(
                self,
                "No Results",
                "No kinetics results available for linearity check."
            )
            return

        # Use stored selection, or all compounds if none stored
        compounds_to_analyze = self._linearity_check_selection if self._linearity_check_selection else list(kinetics_results.keys())
        logger.info(f"Analyzing {len(compounds_to_analyze)} compounds for linearity check")

        # Build selected_items list (only for selected compounds)
        selected_items = []
        a_values = []
        ei_values = []
        ei_unit = None

        for compound_name in compounds_to_analyze:
            if compound_name not in kinetics_results:
                logger.warning(f"Compound {compound_name} not in kinetics results, skipping")
                continue

            compound_data = kinetics_results[compound_name]

            # Get statistics for mean alpha
            stats_data = statistics_results.get(compound_name)
            if not stats_data:
                logger.warning(f"No statistics for {compound_name}, skipping")
                continue

            stats = stats_data.get('statistics', {})
            mean_arrays = stats_data.get('mean_arrays', {})
            sd_arrays = stats_data.get('sd_arrays', {})

            # Get ParsedFile from loaded_compounds (NOT from KineticsResult)
            if compound_name not in self.loaded_compounds:
                logger.warning(f"Compound {compound_name} not in loaded_compounds, skipping")
                continue

            # Get first decay file from this compound
            parsed_files = self.loaded_compounds[compound_name]
            decay_files = [f for f in parsed_files if f.file_type == 'decay']

            if not decay_files:
                logger.warning(f"No decay files for {compound_name}, skipping")
                continue

            decay_file = decay_files[0]  # Use first decay file

            # Get A(lambda) - use method from ParsedFile
            a_lambda = decay_file.get_absorbance_for_replicate(0)

            # Get EI and unit - direct attributes from ParsedFile
            ei_value = decay_file.excitation_intensity if decay_file.excitation_intensity is not None else None
            if ei_unit is None and decay_file.intensity_unit:
                ei_unit = decay_file.intensity_unit

            # Format EI string for plotter (e.g., "50 mW")
            if ei_value is not None and ei_value > 0:
                ei_str = f"{ei_value}"
                if ei_unit:
                    ei_str += f" {ei_unit}"
            else:
                ei_str = ""

            logger.info(f"{compound_name}: file={decay_file.file_path}, A(λ)={a_lambda:.3f}, EI={ei_str}")

            # Track numerical values for CV calculation
            a_values.append(a_lambda)
            if ei_value is not None and ei_value > 0:
                ei_values.append(ei_value)
            else:
                ei_values.append(0)

            # Build item dict - need to include analysis results for variable_study_plotter
            item = {
                'type': 'mean',
                'compound': compound_name,
                'wavelength': compound_data.get('wavelength', 0),
                'classification': compound_data.get('classification', ''),
                'mean_arrays': mean_arrays,
                'sd_arrays': sd_arrays,
                'abs_at_wavelength': f"{a_lambda:.3f}",  # String format for plotter compatibility
                'excitation_intensity': ei_str,  # String format: "value unit" - for variable_study_plotter
                'alpha_mean': stats.get('A_mean', 0),
                'alpha_sd': stats.get('A_sd', 0),
                'analysis_result': stats_data  # Include full statistics for plotter
            }
            selected_items.append(item)

        if not selected_items:
            QMessageBox.warning(
                self,
                "No Data",
                "No valid datasets found for linearity check."
            )
            return

        # Calculate coefficient of variation (CV) for A and EI
        a_array = np.array(a_values)
        ei_array = np.array(ei_values)

        a_cv = (np.std(a_array) / np.mean(a_array) * 100) if np.mean(a_array) > 0 else 0
        ei_cv = (np.std(ei_array) / np.mean(ei_array) * 100) if np.mean(ei_array) > 0 else 0

        logger.info(f"Coefficient of Variation: A(λ) = {a_cv:.1f}%, EI = {ei_cv:.1f}%")

        # Determine which plots to generate
        threshold = 10.0  # 10% CV threshold
        plots_created = []

        # Set busy cursor for all plot generation
        self._set_busy_cursor()

        # Always create combined emission plot
        try:
            self.status_label.setText("Generating combined emission plot...")
            QApplication.processEvents()
            self._on_plot_merged_requested(selected_items)
            plots_created.append("Combined Emission")
        except Exception as e:
            logger.error(f"Failed to create combined emission plot: {e}")

        # Create α vs A plot if A varies
        if a_cv > threshold:
            try:
                self.status_label.setText("Generating Beer-Lambert linearity plot...")
                QApplication.processEvents()

                # Log operation for replay
                operation = {
                    'type': 'linearity',
                    'params': {
                        'check_type': 'beer_lambert',
                        'compounds': compounds_to_analyze
                    }
                }
                self.plot_operations.append(operation)

                fig, regression_stats = plot_alpha_vs_absorption_mpl(selected_items, ei_unit or "")
                plot_id = "Alpha_vs_Absorption"
                title = "α vs (1 - 10^(-A(λ)))"
                self._show_plot(plot_id, title, fig)
                plots_created.append("α vs Absorption")
                logger.info(f"Beer-Lambert linearity: R² = {regression_stats.get('r_squared', 0):.4f}")
            except Exception as e:
                logger.error(f"Failed to create α vs absorption plot: {e}")

        # Create α vs EI plot if EI varies
        if ei_cv > threshold:
            try:
                self.status_label.setText("Generating intensity linearity plot...")
                QApplication.processEvents()

                # Log operation for replay
                operation = {
                    'type': 'linearity',
                    'params': {
                        'check_type': 'excitation_intensity',
                        'compounds': compounds_to_analyze
                    }
                }
                self.plot_operations.append(operation)

                fig, regression_stats = plot_alpha_vs_intensity_mpl(selected_items, ei_unit or "")
                plot_id = "Alpha_vs_Intensity"
                title = "α vs Excitation Intensity"
                self._show_plot(plot_id, title, fig)
                plots_created.append("α vs Intensity")
                logger.info(f"Intensity linearity: R² = {regression_stats.get('r_squared', 0):.4f}")
            except Exception as e:
                logger.error(f"Failed to create α vs intensity plot: {e}")

        # Restore cursor before showing dialogs
        self._restore_cursor()

        # Show summary
        if plots_created:
            msg = f"Linearity check complete!\n\n"
            msg += f"Analyzed {len(selected_items)} compound(s): {', '.join(compounds_to_analyze)}\n\n"
            msg += f"Plots generated:\n" + "\n".join(f"• {p}" for p in plots_created)
            msg += f"\n\nParameter variation:\n• A(λ): CV = {a_cv:.1f}%\n• EI: CV = {ei_cv:.1f}%"
            QMessageBox.information(self, "Linearity Check Complete", msg)
            self.status_label.setText("Linearity check complete - plots generated")
        else:
            QMessageBox.warning(
                self,
                "Insufficient Variation",
                f"Analyzed {len(selected_items)} compound(s), but parameter variation is too low:\n\n"
                f"• A(λ): CV = {a_cv:.1f}% (threshold: {threshold}%)\n"
                f"• EI: CV = {ei_cv:.1f}% (threshold: {threshold}%)\n\n"
                f"At least one parameter must vary by >{threshold}% to perform linearity check.\n\n"
                f"Only the Combined Emission plot was created."
            )
            self.status_label.setText("Linearity check: insufficient parameter variation")

    def _run_surplus_analysis(self):
        """
        Run surplus analysis on selected compounds (automated workflow).

        New Workflow (Session 22):
        1. Auto-detect selected compounds from Tab 1 (Loaded Files)
        2. Auto-run homogeneous analysis if not done yet (in one go!)
        3. Use Replicate 1 ONLY for each compound
        4. Extract tau_T from kinetics results automatically
        5. Use mask_time from preferences (default 6.0 μs)
        6. Run surplus analysis automatically
        7. Display results in Tab 4 (Surplus Results)

        No extra clicks needed!
        """
        from plotting.solis_plotter import SOLISPlotter
        from surplus.surplus_analyzer import analyze_surplus

        # Get selected compounds from Tab 1 (Loaded Files)
        selected_compounds = self.file_browser.get_selected_compounds_for_analysis()

        if not selected_compounds:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select compounds to analyze by checking them in Tab 1 (Loaded Files)."
            )
            return

        logger.info(f"Surplus analysis: {len(selected_compounds)} compounds selected")

        # Check if homogeneous analysis has been run
        kinetics_results = self.results_viewer.kinetics_results

        # Determine which compounds need homogeneous analysis
        compounds_needing_kinetics = []
        for compound_name in selected_compounds.keys():
            if compound_name not in kinetics_results:
                compounds_needing_kinetics.append(compound_name)

        # If any compound needs homogeneous analysis, run it first THEN continue
        if compounds_needing_kinetics:
            response = QMessageBox.question(
                self,
                "Run Homogeneous Analysis First?",
                f"{len(compounds_needing_kinetics)} compound(s) need homogeneous analysis first.\n\n"
                f"Run homogeneous analysis AND surplus in one go?\n\n"
                f"Compounds: {', '.join(compounds_needing_kinetics[:3])}"
                f"{'...' if len(compounds_needing_kinetics) > 3 else ''}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if response == QMessageBox.StandardButton.Yes:
                # Set flag to run surplus after homogeneous completes
                self._pending_surplus_analysis = True
                logger.info("Running homogeneous analysis, will auto-run surplus after")

                # Run homogeneous analysis first
                self._run_analysis('homogeneous')
                return  # Surplus will be triggered by _on_analysis_complete
            else:
                # User declined - can only analyze compounds with existing results
                logger.info("User declined homogeneous analysis - filtering compounds")
                selected_compounds = {k: v for k, v in selected_compounds.items()
                                     if k in kinetics_results}

                if not selected_compounds:
                    QMessageBox.warning(
                        self,
                        "No Results Available",
                        "No compounds have kinetics results.\n\n"
                        "Please run homogeneous analysis first."
                    )
                    return

        # Run the actual surplus analysis
        self._execute_surplus_analysis(selected_compounds)

    def _execute_surplus_analysis(self, selected_compounds=None):
        """Execute surplus analysis on given compounds (internal method)."""
        from surplus.surplus_analyzer import analyze_surplus

        # If no compounds specified, get from Tab 1
        if selected_compounds is None:
            selected_compounds = self.file_browser.get_selected_compounds_for_analysis()
            if not selected_compounds:
                logger.warning("No compounds selected for surplus analysis")
                return

        # Filter to only compounds with kinetics results
        kinetics_results = self.results_viewer.kinetics_results
        selected_compounds = {k: v for k, v in selected_compounds.items()
                             if k in kinetics_results}

        if not selected_compounds:
            QMessageBox.warning(
                self,
                "No Results Available",
                "No compounds have kinetics results.\n\n"
                "Please run homogeneous analysis first."
            )
            return

        # Get mask_time from preferences
        mask_time = self.preferences.get('surplus', {}).get('mask_time_us', 6.0)
        logger.info(f"Using mask_time from preferences: {mask_time:.1f} μs")

        # Run surplus analysis on each compound (Replicate 1 only)
        surplus_results = {}
        failed_compounds = []

        self.status_label.setText(f"Running surplus analysis on {len(selected_compounds)} compounds...")

        for compound_name in selected_compounds.keys():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Surplus analysis: {compound_name}")
                logger.info(f"{'='*60}")

                # Get kinetics results for this compound
                compound_data = kinetics_results.get(compound_name)
                if not compound_data:
                    logger.warning(f"No kinetics results for {compound_name}")
                    failed_compounds.append(compound_name)
                    continue

                # Get Replicate 1 results
                replicate_results = compound_data.get('results', [])
                if not replicate_results or len(replicate_results) < 1:
                    logger.warning(f"No replicate data for {compound_name}")
                    failed_compounds.append(compound_name)
                    continue

                # Use Replicate 1 (index 0)
                rep1_result = replicate_results[0]

                # Extract data
                time = rep1_result.time_experiment_us
                intensity = rep1_result.intensity_raw
                tau_T = rep1_result.parameters.tau_T
                tau_delta = rep1_result.parameters.tau_delta

                logger.info(f"Using Replicate 1 data: {len(time)} points")
                logger.info(f"Extracted tau_T = {tau_T:.3f} μs from kinetics")
                logger.info(f"Extracted tau_delta = {tau_delta:.3f} μs from kinetics")

                # Run surplus analysis
                result = analyze_surplus(
                    time=time,
                    intensity=intensity,
                    mask_time=mask_time,
                    tau_T_guess=tau_T,  # Use kinetics tau_T as initial guess
                    tau_delta_fixed=tau_delta  # Use kinetics tau_delta
                )

                surplus_results[compound_name] = result
                logger.info(f"✓ Surplus analysis complete: {compound_name}")

            except Exception as e:
                logger.error(f"Surplus analysis failed for {compound_name}: {e}", exc_info=True)
                failed_compounds.append(compound_name)

        # Update Tab 4 (Surplus Results) in integrated browser
        if surplus_results:
            self.results_viewer.populate_surplus_results(surplus_results)

            # Auto-show surplus plots for each compound
            for compound_name in surplus_results.keys():
                self._on_surplus_plot_requested(compound_name)

            logger.info(f"Surplus results populated: {len(surplus_results)} compounds, plots auto-opened")

        # Show completion message
        if surplus_results and not failed_compounds:
            QMessageBox.information(
                self,
                "Surplus Analysis Complete",
                f"Surplus analysis completed successfully!\n\n"
                f"Analyzed {len(surplus_results)} compound(s).\n"
                f"Results displayed in Tab 4 (Surplus Results)."
            )
            self.status_label.setText(f"Surplus analysis complete: {len(surplus_results)} compounds")

        elif surplus_results and failed_compounds:
            QMessageBox.warning(
                self,
                "Surplus Analysis Partially Complete",
                f"Surplus analysis completed for {len(surplus_results)} compound(s).\n\n"
                f"Failed for {len(failed_compounds)} compound(s):\n"
                f"{', '.join(failed_compounds[:5])}"
                f"{'...' if len(failed_compounds) > 5 else ''}\n\n"
                f"Check console for details."
            )
            self.status_label.setText(f"Surplus analysis: {len(surplus_results)} OK, {len(failed_compounds)} failed")

        else:
            QMessageBox.critical(
                self,
                "Surplus Analysis Failed",
                f"Surplus analysis failed for all compounds.\n\n"
                f"Check console for details."
            )
            self.status_label.setText("Surplus analysis failed")

    def _on_surplus_plot_requested(self, compound_name: str):
        """Handle request to view surplus analysis plot."""
        from plotting.solis_plotter import SOLISPlotter

        # Get surplus result from integrated browser
        surplus_results = self.results_viewer.surplus_results
        if not surplus_results or compound_name not in surplus_results:
            logger.warning(f"No surplus results found for {compound_name}")
            return

        result = surplus_results[compound_name]

        logger.info(f"Creating surplus plot for {compound_name}")
        self.status_label.setText(f"Creating surplus plot for {compound_name}...")
        self._set_busy_cursor()

        try:
            # Log operation for replay
            operation = {
                'type': 'surplus',
                'params': {'compound': compound_name}
            }
            self.plot_operations.append(operation)

            # Create plot
            plotter = SOLISPlotter()
            fig = plotter.plot_surplus_analysis_mpl(result, log_x=True)

            # Show plot
            plot_id = f"surplus_{compound_name}"
            title = f"Surplus Analysis - {compound_name}"
            self._show_plot(plot_id, title, fig)

            self._restore_cursor()
            self.status_label.setText(f"Surplus plot opened: {compound_name}")

        except Exception as e:
            self._restore_cursor()
            logger.error(f"Failed to create surplus plot for {compound_name}: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Plot Error",
                f"Failed to create surplus plot:\n\n{str(e)}\n\nCheck console for details."
            )
            self.status_label.setText("Surplus plot failed")

    def _run_vesicle_analysis(self):
        """
        Run heterogeneous diffusion analysis on selected decay file.

        Shows parameter dialog, then runs analysis with new heterogeneous code.
        """
        # Get selected file from integrated browser
        selected_compounds = self.file_browser.get_selected_compounds_for_analysis()

        if not selected_compounds:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a compound to analyze by checking it in Tab 1 (Loaded Files)."
            )
            return

        # Use first selected compound, first decay file
        compound_name = list(selected_compounds.keys())[0]
        decay_files = selected_compounds[compound_name]

        if not decay_files:
            QMessageBox.warning(
                self,
                "No Decay Files",
                f"No decay files found for {compound_name}"
            )
            return

        decay_file = decay_files[0]  # Use first file

        # Get data
        time, intensity_replicates = decay_file.get_kinetics_data()
        intensity = intensity_replicates[0]  # Use first replicate

        logger.info(f"Heterogeneous analysis: {compound_name} Rep1 - {decay_file.file_path}")

        # Show parameter dialog
        from gui.heterogeneous_dialog import HeterogeneousDialog
        dialog = HeterogeneousDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            logger.info("Heterogeneous analysis cancelled by user")
            return

        # Get parameters from dialog
        params = dialog.get_parameters()
        logger.info(f"Heterogeneous parameters: {params}")

        # Call the actual analysis method with parameters
        self._execute_vesicle_analysis(compound_name, 1, time, intensity, params)

    # Right-click vesicle analysis removed - use menu only for heterogeneous analysis
    # def _run_vesicle_analysis_direct(self, compound_name: str, replicate_num: int, replicate_data: dict):
    #     """
    #     Run vesicle analysis on specific replicate (called from context menu).
    #     REMOVED: Use menu option instead to configure parameters via dialog.
    #     """
    #     pass

    def _execute_vesicle_analysis(self, compound_name: str, replicate_num: int, time, intensity, params: dict):
        """
        Execute heterogeneous diffusion analysis using new Numba-optimized code.

        Parameters are already obtained from dialog.
        Uses heterogeneous_fitter.py with custom geometry and diffusion parameters.
        """
        from heterogeneous.heterogeneous_fitter import HeterogeneousFitter
        from heterogeneous.heterogeneous_dataclasses import VesicleGeometry, DiffusionParameters
        import numpy as np
        from PyQt6.QtCore import QThread, pyqtSignal

        logger.info(f"Starting heterogeneous analysis: {compound_name} Rep{replicate_num}")
        logger.info(f"Parameters: preset={params['preset']}, fit=[{params['fit_start_us']}, {params['fit_end_us']}]μs")

        # Update status
        self.status_label.setText(f"Running heterogeneous analysis on {compound_name}...")

        # All parameters now come from dialog (not preferences!)
        # Create geometry from dialog parameters
        geometry = VesicleGeometry(
            n_layers=params['n_layers'],
            layer_thickness=1.0,  # Fixed at 1 nm
            membrane_start=params['membrane_start'],
            membrane_end=params['membrane_end'],
            ps_layers=params['ps_layers']
        )

        # Create diffusion parameters from dialog
        # Note: tau_T and tau_delta_water will be fitted, these are just initial values
        base_params = DiffusionParameters(
            D_water=params['D_water'],
            D_lipid=params['D_lipid'],
            tau_delta_lipid=params['tau_delta_lipid'],
            partition_coeff=params['partition_coeff'],
            tau_T=2.0,  # Initial guess, will be fitted
            tau_delta_water=3.7,  # Initial guess, will be fitted
            time_step=1.25e-4,  # Fixed at 0.125 ns
            output_time_step=0.02,  # Fixed at 20 ns
            max_time=params['fit_end_us']  # Use fit range maximum
        )

        # Create worker thread
        class HeterogeneousWorkerNew(QThread):
            """Worker thread for heterogeneous analysis with progress tracking."""
            progress_update = pyqtSignal(str)
            progress_percentage = pyqtSignal(int)  # NEW: Progress percentage (0-100)
            analysis_complete = pyqtSignal(object)  # result object
            error_occurred = pyqtSignal(str)

            def __init__(self, time_exp, signal_exp, geometry, base_params, fit_params):
                super().__init__()
                self.time_exp = time_exp
                self.signal_exp = signal_exp
                self.geometry = geometry
                self.base_params = base_params
                self.fit_params = fit_params

            def run(self):
                try:
                    import time as time_module
                    from heterogeneous.grid_search import GridSearchParams

                    # Create fitter with custom geometry and parameters
                    fitter = HeterogeneousFitter(
                        geometry=self.geometry,
                        base_parameters=self.base_params
                    )

                    # Build custom grid search parameters from dialog settings
                    # Single-step grid search (no two-step refinement)
                    grid_points = self.fit_params['grid_points']
                    total_sims = grid_points * grid_points

                    custom_grid = GridSearchParams.custom(
                        tau_T_range=(self.fit_params['tau_T_min'], self.fit_params['tau_T_max']),
                        tau_delta_W_range=(self.fit_params['tau_w_min'], self.fit_params['tau_w_max']),
                        grid_points=grid_points  # User-defined grid density (e.g., 15 for 15×15)
                    )

                    # Progress callback for grid search (updates every ~10 simulations)
                    def progress_cb(percentage):
                        self.progress_percentage.emit(percentage)

                    # Run fit with custom parameters
                    self.progress_update.emit(f"Running grid search ({total_sims} simulations)...")
                    self.progress_percentage.emit(10)

                    result = fitter.fit(
                        time_exp=self.time_exp,
                        intensity_exp=self.signal_exp,
                        custom_params=custom_grid,  # Use custom params instead of preset!
                        fit_range=(self.fit_params['fit_start_us'], self.fit_params['fit_end_us']),
                        progress_callback=progress_cb  # Real-time progress updates!
                    )

                    self.progress_percentage.emit(90)
                    self.progress_update.emit("Finalizing results...")
                    self.progress_percentage.emit(100)

                    self.analysis_complete.emit(result)

                except Exception as e:
                    logger.error(f"Heterogeneous analysis failed: {e}", exc_info=True)
                    self.error_occurred.emit(str(e))

        # Create and start worker
        worker = HeterogeneousWorkerNew(time, intensity, geometry, base_params, params)
        worker.progress_update.connect(lambda msg: self.status_label.setText(f"Heterogeneous: {msg}"))
        worker.progress_percentage.connect(self._update_progress)
        worker.error_occurred.connect(lambda err: self._on_vesicle_error(err))
        worker.analysis_complete.connect(lambda result: self._on_vesicle_complete(compound_name, replicate_num, result))

        # Start progress tracking
        self._start_progress()
        worker.start()

        # Store worker reference (prevent garbage collection)
        self._vesicle_worker = worker

    def _on_vesicle_complete(self, compound_name: str, replicate_num: int, result):
        """Handle completion of vesicle analysis."""
        from heterogeneous.heterogeneous_plotter_new import HeterogeneousPlotter

        # Stop progress tracking
        self._stop_progress(f"Heterogeneous analysis complete: {compound_name}")

        logger.info(f"Vesicle analysis complete: {compound_name}")

        # Store result in integrated browser
        if not hasattr(self.results_viewer, 'heterogeneous_results'):
            self.results_viewer.heterogeneous_results = {}

        key = f"{compound_name}_Rep{replicate_num}"
        self.results_viewer.heterogeneous_results[key] = result

        # Update Tab 5 (Heterogeneous Results)
        self.results_viewer.populate_heterogeneous_results(self.results_viewer.heterogeneous_results)

        # Create plots using new plotter
        plotter = HeterogeneousPlotter(result)

        # Plot 1: Fit curves with components
        fig_fit = plotter.plot_fit_curves(show_components=True)
        plot_id_fit = f"hetero_fit_{compound_name}_Rep{replicate_num}"
        title_fit = f"Heterogeneous Fit - {compound_name} Rep{replicate_num}"
        self._show_plot(plot_id_fit, title_fit, fig_fit)

        # Plot 2: Chi-square landscape (Figure 4)
        try:
            fig_grid = plotter.plot_figure_4()
            plot_id_grid = f"hetero_grid_{compound_name}_Rep{replicate_num}"
            title_grid = f"Chi-square Landscape - {compound_name} Rep{replicate_num}"
            self._show_plot(plot_id_grid, title_grid, fig_grid)
        except ValueError as e:
            logger.warning(f"Could not generate chi-square landscape: {e}")

        # Show success message
        QMessageBox.information(
            self,
            "Vesicle Analysis Complete",
            f"Heterogeneous vesicle analysis completed successfully!\n\n"
            f"Compound: {compound_name} Rep{replicate_num}\n"
            f"χ²red = {result.reduced_chi_square:.4f}\n\n"
            f"Results displayed in Tab 5 (Heterogeneous Results)."
        )

        self.status_label.setText(f"Vesicle analysis complete: {compound_name}")

    def _on_vesicle_error(self, error_msg: str):
        """Handle vesicle analysis error."""
        # Stop progress tracking
        self._stop_progress("Heterogeneous analysis failed")

        logger.error(f"Vesicle analysis error: {error_msg}")
        QMessageBox.critical(
            self,
            "Vesicle Analysis Error",
            f"Vesicle analysis failed:\n\n{error_msg}\n\nCheck console for details."
        )
        self.status_label.setText("Vesicle analysis failed")

    def _on_heterogeneous_plot_requested(self, key: str):
        """Handle request to view heterogeneous analysis plots (both fit and grid)."""
        from heterogeneous.heterogeneous_plotter_new import HeterogeneousPlotter

        # Get heterogeneous result from integrated browser
        heterogeneous_results = self.results_viewer.heterogeneous_results
        if not heterogeneous_results or key not in heterogeneous_results:
            logger.warning(f"No heterogeneous results found for {key}")
            return

        result = heterogeneous_results[key]

        logger.info(f"Creating heterogeneous plots for {key}")
        self.status_label.setText(f"Creating heterogeneous plots for {key}...")
        self._set_busy_cursor()

        try:
            # Create plotter with result
            plotter = HeterogeneousPlotter(result)

            # Plot 1: Fit curves with components
            # Log operation for replay
            operation_fit = {
                'type': 'heterogeneous_fit',
                'params': {'key': key}
            }
            self.plot_operations.append(operation_fit)

            fig_fit = plotter.plot_fit_curves(show_components=True)
            plot_id_fit = f"hetero_fit_{key}"
            title_fit = f"Heterogeneous Fit - {key}"
            self._show_plot(plot_id_fit, title_fit, fig_fit)

            # Plot 2: Chi-square landscape (Figure 4)
            try:
                # Log operation for replay
                operation_landscape = {
                    'type': 'heterogeneous_landscape',
                    'params': {'key': key}
                }
                self.plot_operations.append(operation_landscape)

                fig_grid = plotter.plot_figure_4()
                plot_id_grid = f"hetero_grid_{key}"
                title_grid = f"Chi-square Landscape - {key}"
                self._show_plot(plot_id_grid, title_grid, fig_grid)
            except ValueError as e:
                logger.warning(f"Could not generate chi-square landscape: {e}")

            self._restore_cursor()
            self.status_label.setText(f"Heterogeneous plots opened: {key}")

        except Exception as e:
            self._restore_cursor()
            logger.error(f"Failed to create heterogeneous plots for {key}: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Plot Error",
                f"Failed to create heterogeneous plots:\n\n{str(e)}\n\nCheck console for details."
            )
            self.status_label.setText("Heterogeneous plots failed")

    def _on_absorption_plot_requested(self, compound_name: str):
        """Handle request to plot single absorption spectrum."""
        from plotting.solis_plotter import SOLISPlotter

        # Get parsed files for this compound
        if not self.loaded_compounds or compound_name not in self.loaded_compounds:
            logger.warning(f"No data found for {compound_name}")
            return

        files = self.loaded_compounds[compound_name]
        abs_files = [f for f in files if f.file_type == 'absorption']

        if not abs_files:
            logger.warning(f"No absorption files found for {compound_name}")
            return

        abs_file = abs_files[0]  # Use first absorption file

        logger.info(f"Creating absorption spectrum plot for {compound_name}")
        self.status_label.setText(f"Creating absorption spectrum for {compound_name}...")
        self._set_busy_cursor()

        try:
            # Extract excitation wavelengths from decay files (if any)
            decay_files = [f for f in files if f.file_type == 'decay']
            excitation_wavelengths = []
            if decay_files:
                excitation_wavelengths = list(set([f.wavelength for f in decay_files if f.wavelength]))

            # Log operation for replay
            operation = {
                'type': 'absorption_single',
                'params': {
                    'compound': compound_name,
                    'excitation_wavelengths': excitation_wavelengths
                }
            }
            self.plot_operations.append(operation)

            # Create plot
            plotter = SOLISPlotter()
            fig = plotter.plot_absorption_spectrum_mpl(
                abs_file,
                excitation_wavelengths=excitation_wavelengths if excitation_wavelengths else None
            )

            # Show plot
            plot_id = f"abs_{compound_name}"
            title = f"Absorption Spectrum - {compound_name}"
            self._show_plot(plot_id, title, fig)

            self._restore_cursor()
            self.status_label.setText(f"Absorption spectrum plot opened: {compound_name}")

        except Exception as e:
            self._restore_cursor()
            logger.error(f"Failed to create absorption plot for {compound_name}: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Plot Error",
                f"Failed to create absorption spectrum:\n\n{str(e)}\n\nCheck console for details."
            )
            self.status_label.setText("Absorption plot failed")

    def _on_absorption_merged_requested(self, compound_names: List[str]):
        """Handle request to plot merged absorption spectra."""
        from plotting.solis_plotter import SOLISPlotter

        if not self.loaded_compounds:
            logger.warning("No data loaded")
            return

        # Collect absorption files and excitation wavelengths
        parsed_files = []
        excitation_wavelengths = {}

        for compound_name in compound_names:
            if compound_name not in self.loaded_compounds:
                continue

            files = self.loaded_compounds[compound_name]
            abs_files = [f for f in files if f.file_type == 'absorption']

            if abs_files:
                parsed_files.append(abs_files[0])

                # Extract excitation wavelengths from decay files
                decay_files = [f for f in files if f.file_type == 'decay']
                if decay_files:
                    ex_wls = list(set([f.wavelength for f in decay_files if f.wavelength]))
                    if ex_wls:
                        excitation_wavelengths[compound_name] = ex_wls

        if not parsed_files:
            logger.warning("No absorption files found for selected compounds")
            QMessageBox.warning(
                self,
                "No Data",
                "No absorption files found for the selected compounds."
            )
            return

        logger.info(f"Creating merged absorption spectra plot with {len(parsed_files)} spectra")
        self.status_label.setText(f"Creating merged absorption plot ({len(parsed_files)} compounds)...")
        self._set_busy_cursor()

        try:
            # Log operation for replay
            operation = {
                'type': 'absorption_merged',
                'params': {
                    'compounds': compound_names,
                    'excitation_wavelengths': excitation_wavelengths
                }
            }
            self.plot_operations.append(operation)

            # Create plot
            plotter = SOLISPlotter()
            fig = plotter.plot_merged_absorption_spectra_mpl(
                parsed_files,
                excitation_wavelengths=excitation_wavelengths if excitation_wavelengths else None
            )

            # Show plot
            if len(compound_names) > 3:
                plot_id = f"abs_merged_{len(compound_names)}"
                title = f"Absorption Spectra ({len(compound_names)} compounds)"
            else:
                plot_id = "abs_merged_" + "_".join(compound_names)
                title = "Absorption Spectra: " + ", ".join(compound_names)

            self._show_plot(plot_id, title, fig)

            self._restore_cursor()
            self.status_label.setText(f"Merged absorption plot opened ({len(parsed_files)} compounds)")

        except Exception as e:
            self._restore_cursor()
            logger.error(f"Failed to create merged absorption plot: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Plot Error",
                f"Failed to create merged absorption spectra:\n\n{str(e)}\n\nCheck console for details."
            )
            self.status_label.setText("Merged absorption plot failed")

    def _show_preview_plot(self, compound_name: str, replicate_num: int, replicate_data: dict):
        """
        Display preview plot for selected replicate.

        Parameters
        ----------
        compound_name : str
            Name of the compound
        replicate_num : int
            Replicate number (1-indexed)
        replicate_data : dict
            Dictionary with 'time', 'intensity', 'file_path', etc.
        """
        try:
            # Run SNR analysis
            from core.snr_analyzer import SNRAnalyzer
            snr_analyzer = SNRAnalyzer()

            time = replicate_data['time']
            intensity = replicate_data['intensity']
            snr_result = snr_analyzer.analyze_snr(time, intensity)

            # Create plot ID
            plot_id = f"{compound_name}_Rep{replicate_num}"

            # Create or reuse plot viewer
            if plot_id not in self.plot_viewers:
                viewer = PlotViewerWidget(self, plot_title=f"Preview: {plot_id}")
                viewer.window_closed.connect(lambda: self._on_plot_closed(plot_id))
                viewer.mask_parameters_changed.connect(self._on_mask_corrected)
                self.plot_viewers[plot_id] = viewer

                # Preview plots are floating only - NOT added to Plots menu
                # User can manually set mask time in the preview window

            viewer = self.plot_viewers[plot_id]

            # Extract spike mask from SNRResult (spike_region is a dict with 'start' and 'end' indices)
            spike_mask = None
            mask_end_time = None
            if snr_result.spike_region is not None:
                spike_start_idx = snr_result.spike_region['start']
                spike_end_idx = snr_result.spike_region['end']
                # Create boolean mask: True=signal, False=spike
                spike_mask = np.arange(len(time)) > spike_end_idx
                # Get mask endpoint time
                if spike_end_idx < len(time):
                    mask_end_time = time[spike_end_idx]
                else:
                    mask_end_time = time[-1]

            # Update plot with preview data
            viewer.update_plot({
                'file_type': 'decay',
                'preview_mode': True,
                'time': time,
                'intensity': intensity,
                'spike_mask': spike_mask,
                'mask_end_time': mask_end_time,
                'snr_result': snr_result,
                'display_label': f'Replicate {replicate_num}',
                'file_path': replicate_data['file_path'],
                'compound_name': compound_name,
                'replicate_num': replicate_num
            })

            # Show window (don't steal focus)
            viewer.show()
            # Keep main window in focus
            self.raise_()
            self.activateWindow()

            self.status_label.setText(f"Preview plot opened: {plot_id}")
            logger.info(f"Preview plot opened: {plot_id}")

        except Exception as e:
            logger.error(f"Failed to create preview plot: {e}")
            QMessageBox.critical(
                self,
                "Plot Error",
                f"Failed to create preview plot:\n\n{str(e)}"
            )

    def _on_mask_corrected(self, params: dict):
        """
        Handle mask correction from preview plot.

        Parameters
        ----------
        params : dict
            Contains: mask_end_time, apply_to_all, compound_name, replicate_num
        """
        mask_time = params['mask_end_time']
        apply_to_all = params['apply_to_all']
        compound = params['compound_name']
        replicate = params['replicate_num']

        if apply_to_all:
            # Store correction for ALL replicates in this compound
            self.mask_corrections[compound] = mask_time
            logger.info(f"Mask correction stored for {compound} (all replicates): {mask_time:.4f} μs")
            self.status_label.setText(
                f"Mask correction applied to all replicates of {compound}: {mask_time:.4f} μs"
            )
        else:
            # Store for specific replicate only
            key = f"{compound}_Rep{replicate}"
            self.mask_corrections[key] = mask_time
            logger.info(f"Mask correction stored for {key}: {mask_time:.4f} μs")
            self.status_label.setText(
                f"Mask correction applied to {key}: {mask_time:.4f} μs"
            )

    def _toggle_plot(self, plot_id: str, checked: bool):
        """Toggle plot window visibility."""
        if plot_id in self.plot_viewers:
            viewer = self.plot_viewers[plot_id]
            if checked:
                viewer.show()
                viewer.raise_()
                viewer.activateWindow()
            else:
                viewer.hide()

    def _on_plot_closed(self, plot_id: str):
        """Handle plot window close event."""
        # Plot closed - keep viewer in dictionary for potential reuse
        pass

    def _toggle_select_all(self):
        """Toggle selection of all compounds in current tab (highlight-based selection)."""
        # Load icons
        icon_dir = Path(__file__).parent / 'icons'

        # Determine which tab is active
        if hasattr(self.results_viewer, 'kinetics_tree') and 'Kinetics' in self.results_viewer.tab_indices:
            tree = self.results_viewer.kinetics_tree
            tab_name = "Kinetics Results"
        else:
            # Default to decay files in browser
            tree = self.file_browser.tree
            tab_name = "Decay Files"

        # Check CURRENT state by examining selected items
        items = []
        for i in range(self.file_browser.decay_section.childCount()):
            item = self.file_browser.decay_section.child(i)
            items.append(item)

        all_selected = len(tree.selectedItems()) == len(items) if items else False

        # Toggle: if all selected → clear selection, otherwise → select all
        if all_selected:
            tree.clearSelection()
            # Update button to "Select All Decays"
            select_icon = QIcon(str(icon_dir / 'check_box_outline_blank_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg'))
            self.select_all_action.setIcon(select_icon)
            self.select_all_action.setText("Select All Decays")
            self.select_all_action.setToolTip("Select all decay compounds in browser")
            self.status_label.setText("All decay compounds deselected")
        else:
            # Select all items
            for item in items:
                item.setSelected(True)
            # Update button to "Deselect All Decays"
            deselect_icon = QIcon(str(icon_dir / 'check_box_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.svg'))
            self.select_all_action.setIcon(deselect_icon)
            self.select_all_action.setText("Deselect All Decays")
            self.select_all_action.setToolTip("Deselect all decay compounds in browser")
            self.status_label.setText("All decay compounds selected")

    def _on_toolbar_individual_plots(self):
        """Handle Individual Plots button click from toolbar."""
        # Delegate to integrated browser's method
        self._on_view_plot_from_kinetics()

    def _on_toolbar_plot_merged(self):
        """Handle Plot Merged button click from toolbar."""
        # Delegate to integrated browser's method
        self._on_plot_merged_from_kinetics()

    def closeEvent(self, event):
        """Handle application close event - full cleanup."""
        logger.info("Application closing - performing cleanup")

        # Stop analysis if running
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.stop()
            self.analysis_worker.wait()

        # Close browser dock
        if self.data_browser_dock:
            self.data_browser_dock.close()

        # Destroy all plot data and close viewers
        for viewer in self.plot_viewers.values():
            viewer.destroy_plot()  # Clear all data, cache, and temp files
            viewer.close()

        logger.info("Application cleanup complete")
        event.accept()
    def _on_plot_combined_emission_requested(self, selected_items: list):
        """Handle combined emission plot request from Variable Study widget."""
        # This is essentially a merged plot - use existing functionality
        self._on_plot_merged_requested(selected_items)

    def _on_plot_alpha_vs_absorption_requested(self, selected_items: list, ei_unit: str):
        """Handle α vs absorption plot request from Variable Study widget."""
        from plotting.variable_study_plotter import plot_alpha_vs_absorption_mpl

        try:
            fig, regression_stats = plot_alpha_vs_absorption_mpl(selected_items, ei_unit)

            # Show in plot viewer
            plot_id = "Alpha_vs_Absorption"
            title = "α vs (1 - 10^(-A(λ)))"
            self._show_plot(plot_id, title, fig)

            # Update regression stats in Variable Study widget
            if self.variable_study_widget and regression_stats:
                self.variable_study_widget.update_regression_stats(regression_stats, 'absorption')

        except Exception as e:
            QMessageBox.critical(
                self,
                "Plot Error",
                f"Failed to create α vs absorption plot:\n{str(e)}"
            )
            logger.exception("Error creating α vs absorption plot")

    def _on_plot_alpha_vs_intensity_requested(self, selected_items: list, ei_unit: str):
        """Handle α vs intensity plot request from Variable Study widget."""
        from plotting.variable_study_plotter import plot_alpha_vs_intensity_mpl

        try:
            fig, regression_stats = plot_alpha_vs_intensity_mpl(selected_items, ei_unit)

            # Show in plot viewer
            plot_id = "Alpha_vs_Intensity"
            title = "α vs Excitation Intensity"
            self._show_plot(plot_id, title, fig)

            # Update regression stats in Variable Study widget
            if self.variable_study_widget and regression_stats:
                self.variable_study_widget.update_regression_stats(regression_stats, 'intensity')

        except Exception as e:
            QMessageBox.critical(
                self,
                "Plot Error",
                f"Failed to create α vs intensity plot:\n{str(e)}"
            )
            logger.exception("Error creating α vs intensity plot")


class HelpDialog(QDialog):
    """Help dialog that displays content from help.json file."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SOLIS Help")
        self.setMinimumSize(700, 600)
        self.resize(800, 700)

        # Load help content from JSON
        help_data = self._load_help_json()

        # Create layout
        layout = QVBoxLayout()

        # Create text browser for displaying help content
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        # Style for help text browser (white background, readable font)
        self.text_browser.setStyleSheet(
            "QTextBrowser { background-color: white; font-size: 11pt; padding: 10px; }"
        )

        # Format and display help content
        if help_data:
            html_content = self._format_help_as_html(help_data)
            self.text_browser.setHtml(html_content)
        else:
            self.text_browser.setPlainText(
                "Help file not found.\n\n"
                "Please ensure 'help.json' exists in the SOLIS root directory."
            )

        layout.addWidget(self.text_browser)

        # Add Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.setDefault(True)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _load_help_json(self) -> dict:
        """Load help content from help.json file."""
        try:
            help_file_path = Path(__file__).parent / 'help.json'
            if help_file_path.exists():
                with open(help_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Help file not found at: {help_file_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load help.json: {e}")
            return None

    def _format_help_as_html(self, help_data: dict) -> str:
        """Convert help JSON data to formatted HTML."""
        html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                }}
                h1 {{
                    color: #FF8C00;
                    border-bottom: 3px solid #FF8C00;
                    padding-bottom: 8px;
                    margin-bottom: 20px;
                }}
                h2 {{
                    color: #FF8C00;
                    margin-top: 30px;
                    margin-bottom: 12px;
                    border-bottom: 2px solid #FFE4B5;
                    padding-bottom: 5px;
                }}
                p {{
                    margin: 10px 0;
                }}
                ul {{
                    margin: 8px 0;
                }}
                li {{
                    margin: 5px 0;
                }}
                strong {{
                    color: #FF8C00;
                }}
                code {{
                    background-color: #f5f5f5;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                    font-size: 10pt;
                }}
                .version {{
                    color: #666;
                    font-size: 10pt;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>{help_data.get('title', 'SOLIS Help')}</h1>
            <p class="version">Version: {help_data.get('version', '1.0.0')}</p>
        """

        # Add each section
        for section in help_data.get('sections', []):
            title = section.get('title', 'Untitled')
            content = section.get('content', '')

            # Convert content to HTML (preserve line breaks, bold text, bullets)
            content_html = self._convert_content_to_html(content)

            html += f"""
            <h2>{title}</h2>
            {content_html}
            """

        html += """
        </body>
        </html>
        """

        return html

    def _convert_content_to_html(self, content: str) -> str:
        """Convert plain text content to HTML with basic formatting."""
        # Split into paragraphs
        paragraphs = content.split('\n\n')

        html_parts = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if it's a bullet list (starts with • or -)
            if para.startswith('•') or para.startswith('-'):
                lines = para.split('\n')
                html_parts.append('<ul>')
                for line in lines:
                    line = line.strip()
                    if line.startswith('• '):
                        line = line[2:]
                    elif line.startswith('- '):
                        line = line[2:]

                    # Handle bold text (**text**)
                    line = self._format_bold(line)

                    html_parts.append(f'<li>{line}</li>')
                html_parts.append('</ul>')

            # Check if it's a numbered list
            elif para[0].isdigit() and '. ' in para:
                lines = para.split('\n')
                html_parts.append('<ol>')
                for line in lines:
                    line = line.strip()
                    # Remove leading number and dot
                    if '. ' in line:
                        line = line.split('. ', 1)[1] if '. ' in line else line

                    # Handle bold text
                    line = self._format_bold(line)

                    html_parts.append(f'<li>{line}</li>')
                html_parts.append('</ol>')

            else:
                # Regular paragraph
                # Replace line breaks within paragraph with <br>
                para = para.replace('\n', '<br>')

                # Handle bold text (**text**)
                para = self._format_bold(para)

                html_parts.append(f'<p>{para}</p>')

        return '\n'.join(html_parts)

    def _format_bold(self, text: str) -> str:
        """Convert **text** to <strong>text</strong>."""
        import re
        # Match **text** and convert to <strong>text</strong>
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        return text


def main():
    """Launch SOLIS application."""
    # Create QApplication first (required for splash screen)
    app = QApplication(sys.argv)
    app.setApplicationName("SOLIS")
    app.setOrganizationName("USP-IQ")
    app.setOrganizationDomain("usp.br")

    # Apply global flat stylesheet (removes all shadows from buttons, spinboxes, etc.)
    app.setStyleSheet(get_global_flat_style())

    # macOS-specific: Ensure native menu bar integration
    if sys.platform == 'darwin':
        app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, False)

    # Show splash screen IMMEDIATELY before any heavy imports
    from gui.splash_screen import SOLISSplashScreen
    splash = SOLISSplashScreen()
    splash.show()
    splash.showMessage("Starting SOLIS...")
    app.processEvents()

    # Progress update helper
    def update_progress(value, message):
        splash.setProgress(value)
        splash.showMessage(message)
        app.processEvents()

    # All imports already done at module level, so just show progress
    update_progress(30, "Loading modules...")

    # Create main window
    update_progress(60, "Initializing GUI components...")
    window = SOLISMainWindow()

    # Ready
    update_progress(100, "Ready!")
    app.processEvents()

    # Show main window and close splash
    window.show()
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
