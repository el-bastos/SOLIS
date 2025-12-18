#!/usr/bin/env python3
"""
Dataset Classification Dialog for SOLIS
Allows user to classify dataset structure before analysis
"""

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QButtonGroup, QPushButton, QGroupBox, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class DatasetClassificationDialog(QDialog):
    """
    Dialog for user to classify dataset structure.
    Shows preview plot and asks about lag/spike/signal structure.
    """

    classification_complete = pyqtSignal(str, bool)  # dataset_type, apply_to_all

    def __init__(self, compound_name, time_data, intensity_data, parent=None):
        """
        Initialize classification dialog.

        Args:
            compound_name: Name of the compound
            time_data: Time array (first replicate)
            intensity_data: Intensity array (first replicate)
            parent: Parent widget
        """
        super().__init__(parent)
        self.compound_name = compound_name
        self.time_data = time_data
        self.intensity_data = intensity_data
        self.selected_type = None
        self.apply_to_all = False

        self.setWindowTitle(f"Classify Dataset Structure - {compound_name}")
        self.setMinimumSize(900, 700)
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel(f"<b>Dataset Classification: {self.compound_name}</b>")
        title_font = QFont()
        title_font.setPointSize(12)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Instruction
        instruction = QLabel(
            "Please examine the preview plot below and classify the dataset structure.\n"
            "This will determine how SOLIS processes the timeline and masks artifacts."
        )
        instruction.setWordWrap(True)
        instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instruction)

        # Preview plot
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Plot the preview
        self.plot_preview()

        # Classification options
        group_box = QGroupBox("What does this dataset contain?")
        group_layout = QVBoxLayout()

        self.button_group = QButtonGroup(self)

        # Option (a): Lag + Spike + Signal
        self.radio_lag_spike = QRadioButton(
            "(a) Lag + Spike + Signal (starting at t=0)\n"
            "    → Will remove lag, shift timeline to t=0, mask spike region"
        )
        self.button_group.addButton(self.radio_lag_spike, 0)
        group_layout.addWidget(self.radio_lag_spike)

        # Option (b): Spike + Signal
        self.radio_spike_only = QRadioButton(
            "(b) Spike + Signal (no lag, starting at t=0)\n"
            "    → Will keep timeline as-is, mask spike region only"
        )
        self.button_group.addButton(self.radio_spike_only, 1)
        group_layout.addWidget(self.radio_spike_only)

        # Option (c): Clean signal
        self.radio_clean = QRadioButton(
            "(c) Signal only (clean, starting at t=0)\n"
            "    → No timeline shifting, no masking (fit all points)"
        )
        self.button_group.addButton(self.radio_clean, 2)
        group_layout.addWidget(self.radio_clean)

        # Option (d): Preprocessed
        self.radio_preprocessed = QRadioButton(
            "(d) Signal only (lag/spike already removed by user)\n"
            "    → No timeline shifting, NaN masking only (preserves your timeline)"
        )
        self.button_group.addButton(self.radio_preprocessed, 3)
        group_layout.addWidget(self.radio_preprocessed)

        # Auto-select preprocessed if NaN detected
        if np.any(np.isnan(self.intensity_data)):
            self.radio_preprocessed.setChecked(True)
        else:
            self.radio_lag_spike.setChecked(True)  # Default to most common case

        group_box.setLayout(group_layout)
        layout.addWidget(group_box)

        # Buttons
        button_layout = QHBoxLayout()

        self.apply_all_btn = QPushButton("Apply to All Compounds")
        self.apply_all_btn.setToolTip("Use this classification for all compounds in this session")
        self.apply_all_btn.clicked.connect(self.on_apply_to_all)

        self.ok_btn = QPushButton("OK - Classify This Compound")
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self.on_ok)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.apply_all_btn)
        button_layout.addWidget(self.ok_btn)

        layout.addLayout(button_layout)

    def plot_preview(self):
        """Plot the data preview."""
        self.ax.clear()

        # Plot intensity
        self.ax.plot(self.time_data, self.intensity_data, 'o-', markersize=3, linewidth=1, color='gray', label='Replicate 1')

        # Mark NaN regions if present
        nan_mask = np.isnan(self.intensity_data)
        if np.any(nan_mask):
            nan_indices = np.where(nan_mask)[0]
            if len(nan_indices) > 0:
                self.ax.axvspan(
                    self.time_data[nan_indices[0]],
                    self.time_data[nan_indices[-1]],
                    alpha=0.3, color='red', label='NaN region (user removed)'
                )

        self.ax.set_xlabel('Time (μs)')
        self.ax.set_ylabel('Intensity (photon counts)')
        self.ax.set_title(f'Preview: {self.compound_name} (Replicate 1)')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        # Use logarithmic x-axis for time
        self.ax.set_xscale('log')

        self.figure.tight_layout()
        self.canvas.draw()

    def get_selected_type(self):
        """Get the selected dataset type string."""
        checked_id = self.button_group.checkedId()

        if checked_id == 0:
            return 'lag_spike'
        elif checked_id == 1:
            return 'spike_only'
        elif checked_id == 2:
            return 'clean_signal'
        elif checked_id == 3:
            return 'preprocessed'
        else:
            return 'auto'  # Fallback

    def on_ok(self):
        """Handle OK button - classify this compound only."""
        self.selected_type = self.get_selected_type()
        self.apply_to_all = False
        self.classification_complete.emit(self.selected_type, False)
        self.accept()

    def on_apply_to_all(self):
        """Handle Apply to All button - use for all compounds."""
        self.selected_type = self.get_selected_type()
        self.apply_to_all = True
        self.classification_complete.emit(self.selected_type, True)
        self.accept()


def classify_dataset(compound_name, time_data, intensity_data, parent=None):
    """
    Convenience function to show classification dialog and get result.

    Args:
        compound_name: Name of compound
        time_data: Time array
        intensity_data: Intensity array
        parent: Parent widget

    Returns:
        (dataset_type, apply_to_all): Selected classification and whether to apply to all
    """
    dialog = DatasetClassificationDialog(compound_name, time_data, intensity_data, parent)
    result = dialog.exec()

    if result == QDialog.DialogCode.Accepted:
        return dialog.selected_type, dialog.apply_to_all
    else:
        return None, False
