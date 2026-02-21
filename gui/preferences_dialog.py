#!/usr/bin/env python3
"""
Preferences Dialog - User settings for SOLIS analysis

Contains SNR thresholds and surplus analysis parameters.
Plot appearance settings are in PlotAppearanceDialog (right-click on any plot).
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QPushButton, QHBoxLayout, QGroupBox, QLabel, QWidget
)
from PyQt6.QtCore import Qt
from gui.stylesheets import INFO_LABEL_STYLE


class PreferencesDialog(QDialog):
    """Dialog for editing user preferences (SNR thresholds, surplus settings)."""

    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)

        # Default settings
        if current_settings is None:
            current_settings = {
                'snr_thresholds': {
                    'homogeneous': 5.0,
                    'heterogeneous': 50.0
                },
                'surplus': {
                    'mask_time_us': 6.0
                },
                'heterogeneous_vesicle': {
                    # Geometry parameters
                    'total_time_us': 100.0,
                    'bin_ns': 20.0,
                    'dx_nm': 1.0,
                    'dt_ns': 0.125,
                    'N': 400,
                    'lipid_start': 36,
                    'lipid_thickness': 4,
                    'gen_shells': '37, 38',
                    # Physical parameters
                    'tau_L_us': 14.0,
                    'S': 3.5,
                    'Dw_cm2s': 2e-5,
                    'Dl_cm2s': 1e-5,
                    # Fitting parameters
                    'tau_T_min': 1.5,
                    'tau_T_max': 2.5,
                    'tau_w_min': 3.5,
                    'tau_w_max': 4.5,
                    'grid_points': 15,
                    'fit_start': 0.3,
                    'fit_end': 100.0
                }
            }

        self.settings = current_settings.copy()
        self._setup_ui()

    def _setup_ui(self):
        """Setup dialog UI."""
        main_layout = QVBoxLayout(self)

        # --- SNR Thresholds Group ---
        threshold_group = QGroupBox("SNR Thresholds (Linear Ratio)")
        threshold_layout = QFormLayout()

        info_label = QLabel(
            "Set minimum SNR thresholds for analysis modes.\n"
            "Replicates below threshold will be excluded."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(INFO_LABEL_STYLE)
        threshold_layout.addRow(info_label)

        self.homogeneous_spin = QDoubleSpinBox()
        self.homogeneous_spin.setRange(1.0, 1000.0)
        self.homogeneous_spin.setValue(self.settings['snr_thresholds']['homogeneous'])
        self.homogeneous_spin.setDecimals(1)
        self.homogeneous_spin.setSuffix(":1")
        self.homogeneous_spin.setToolTip("Minimum SNR for homogeneous analysis (default: 5:1)")
        threshold_layout.addRow("Homogeneous:", self.homogeneous_spin)

        self.heterogeneous_spin = QDoubleSpinBox()
        self.heterogeneous_spin.setRange(1.0, 1000.0)
        self.heterogeneous_spin.setValue(self.settings['snr_thresholds']['heterogeneous'])
        self.heterogeneous_spin.setDecimals(1)
        self.heterogeneous_spin.setSuffix(":1")
        self.heterogeneous_spin.setToolTip("Minimum SNR for heterogeneous analysis (default: 50:1)")
        threshold_layout.addRow("Heterogeneous:", self.heterogeneous_spin)

        threshold_group.setLayout(threshold_layout)
        main_layout.addWidget(threshold_group)

        # --- Surplus Analysis Group ---
        surplus_group = QGroupBox("Surplus Analysis Parameters")
        surplus_layout = QFormLayout()

        self.mask_time_spin = QDoubleSpinBox()
        self.mask_time_spin.setRange(0.1, 100.0)
        self.mask_time_spin.setValue(self.settings['surplus']['mask_time_us'])
        self.mask_time_spin.setDecimals(1)
        self.mask_time_spin.setSuffix(" \u03bcs")
        self.mask_time_spin.setToolTip("Time point for late-time fitting in surplus analysis (default: 6.0 \u03bcs)")
        surplus_layout.addRow("Mask Time:", self.mask_time_spin)

        surplus_group.setLayout(surplus_layout)
        main_layout.addWidget(surplus_group)

        # Note about plot appearance
        note = QLabel("Plot appearance: right-click any plot \u2192 Plot Appearance...")
        note.setStyleSheet(INFO_LABEL_STYLE)
        note.setWordWrap(True)
        main_layout.addWidget(note)

        main_layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()

        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        button_layout.addWidget(reset_button)

        button_layout.addStretch()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

        self.setMinimumWidth(380)
        self.setMinimumHeight(320)

    def _reset_defaults(self):
        """Reset all settings to default values."""
        self.homogeneous_spin.setValue(5.0)
        self.heterogeneous_spin.setValue(50.0)
        self.mask_time_spin.setValue(6.0)

    def get_settings(self):
        """
        Get current settings.

        Returns
        -------
        dict
            Dictionary with 'snr_thresholds', 'surplus', and 'plot_style' keys
        """
        return {
            'snr_thresholds': {
                'homogeneous': self.homogeneous_spin.value(),
                'heterogeneous': self.heterogeneous_spin.value()
            },
            'surplus': {
                'mask_time_us': self.mask_time_spin.value()
            },
            'plot_style': self.settings.get('plot_style', {})
        }

    def get_thresholds(self):
        """Backward compatibility: get SNR thresholds only."""
        return self.get_settings()['snr_thresholds']
