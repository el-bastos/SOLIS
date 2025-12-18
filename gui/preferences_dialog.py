#!/usr/bin/env python3
"""
Preferences Dialog - User settings for SOLIS analysis
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QPushButton, QHBoxLayout, QGroupBox, QLabel, QSpinBox,
    QLineEdit, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt
from gui.stylesheets import INFO_LABEL_STYLE


class PreferencesDialog(QDialog):
    """Dialog for editing user preferences."""

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
                    'total_time_us': 100.0,  # Extended to cover typical experimental range (~82 us)
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
                    'fit_start': 0.3,  # Start after spike region
                    'fit_end': 100.0   # Match total_time_us
                }
            }

        self.settings = current_settings.copy()
        self._setup_ui()

    def _setup_ui(self):
        """Setup dialog UI."""
        # Create scroll area for all preferences
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Container widget for scroll area
        container = QWidget()
        layout = QVBoxLayout(container)

        # Main dialog layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)

        # SNR Thresholds Group
        threshold_group = QGroupBox("SNR Thresholds (Linear Ratio)")
        threshold_layout = QFormLayout()

        # Info label
        info_label = QLabel(
            "Set minimum SNR thresholds for analysis modes.\n"
            "Replicates below threshold will be excluded."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(INFO_LABEL_STYLE)
        threshold_layout.addRow(info_label)

        # Homogeneous threshold
        self.homogeneous_spin = QDoubleSpinBox()
        self.homogeneous_spin.setRange(1.0, 1000.0)
        self.homogeneous_spin.setValue(self.settings['snr_thresholds']['homogeneous'])
        self.homogeneous_spin.setDecimals(1)
        self.homogeneous_spin.setSuffix(":1")
        self.homogeneous_spin.setToolTip("Minimum SNR for homogeneous analysis (default: 5:1)")
        threshold_layout.addRow("Homogeneous:", self.homogeneous_spin)

        # Heterogeneous threshold
        self.heterogeneous_spin = QDoubleSpinBox()
        self.heterogeneous_spin.setRange(1.0, 1000.0)
        self.heterogeneous_spin.setValue(self.settings['snr_thresholds']['heterogeneous'])
        self.heterogeneous_spin.setDecimals(1)
        self.heterogeneous_spin.setSuffix(":1")
        self.heterogeneous_spin.setToolTip("Minimum SNR for heterogeneous analysis (default: 50:1)")
        threshold_layout.addRow("Heterogeneous:", self.heterogeneous_spin)

        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)

        # Surplus Analysis Group
        surplus_group = QGroupBox("Surplus Analysis Parameters")
        surplus_layout = QFormLayout()

        # Mask time
        self.mask_time_spin = QDoubleSpinBox()
        self.mask_time_spin.setRange(0.1, 100.0)
        self.mask_time_spin.setValue(self.settings['surplus']['mask_time_us'])
        self.mask_time_spin.setDecimals(1)
        self.mask_time_spin.setSuffix(" μs")
        self.mask_time_spin.setToolTip("Time point for late-time fitting in surplus analysis (default: 6.0 μs)")
        surplus_layout.addRow("Mask Time:", self.mask_time_spin)

        surplus_group.setLayout(surplus_layout)
        layout.addWidget(surplus_group)

        # Heterogeneous Vesicle Analysis - REMOVED
        # All parameters now in HeterogeneousDialog (3-tab popup before analysis)
        # No longer in Preferences - use Analyze → Heterogeneous Analysis menu

        # Set container as scroll area widget
        scroll.setWidget(container)

        # Buttons
        button_layout = QHBoxLayout()

        # Reset to defaults button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        button_layout.addWidget(reset_button)

        button_layout.addStretch()

        # OK/Cancel buttons
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

        # Set dialog size
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)

    def _reset_defaults(self):
        """Reset all settings to default values."""
        # SNR thresholds
        self.homogeneous_spin.setValue(5.0)
        self.heterogeneous_spin.setValue(50.0)

        # Surplus
        self.mask_time_spin.setValue(6.0)

        # Heterogeneous vesicle parameters removed - now in HeterogeneousDialog

    def get_settings(self):
        """
        Get current settings.

        Returns
        -------
        dict
            Dictionary with 'snr_thresholds' and 'surplus' keys
        """
        return {
            'snr_thresholds': {
                'homogeneous': self.homogeneous_spin.value(),
                'heterogeneous': self.heterogeneous_spin.value()
            },
            'surplus': {
                'mask_time_us': self.mask_time_spin.value()
            }
            # heterogeneous_vesicle removed - now in HeterogeneousDialog
        }

    def get_thresholds(self):
        """Backward compatibility: get SNR thresholds only."""
        return self.get_settings()['snr_thresholds']
