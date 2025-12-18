"""
Heterogeneous Analysis Parameter Dialog

Dialog for configuring heterogeneous diffusion model parameters
before running analysis.
"""

import logging
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox, QFormLayout,
    QDialogButtonBox, QTabWidget, QWidget, QLineEdit
)
from PyQt6.QtCore import Qt
from gui.stylesheets import INFO_LABEL_STYLE_SMALL, INFO_LABEL_STYLE_WITH_MARGIN

logger = logging.getLogger(__name__)


class HeterogeneousDialog(QDialog):
    """Dialog for heterogeneous analysis parameter configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Heterogeneous Analysis Settings")
        self.setModal(True)
        self.setMinimumWidth(500)

        # Initialize parameters with defaults (all in dialog, not preferences)
        self.parameters = {
            # Fitting parameters
            'preset': 'medium',
            'fit_start_us': 0.3,
            'fit_end_us': 30.0,
            # Default ranges optimized for typical photosensitizers
            'tau_T_min': 1.5,     # Typical triplet lifetime lower bound
            'tau_T_max': 2.5,     # Typical triplet lifetime upper bound
            'tau_w_min': 3.5,     # tau_delta_water lower bound
            'tau_w_max': 4.5,     # tau_delta_water upper bound
            'grid_points': 15,    # 15×15 = 225 coarse grid points (good balance)
            # Physical parameters (fixed)
            'D_water': 2.0e-5,
            'D_lipid': 1.0e-5,
            'tau_delta_lipid': 14.0,
            'partition_coeff': 3.5,
            # Geometry
            'n_layers': 400,
            'membrane_start': 36,
            'membrane_end': 39,
            'ps_layers': (37, 38)
        }

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI with tabs."""
        layout = QVBoxLayout(self)

        # Create tab widget
        tabs = QTabWidget()

        # Tab 1: Fitting Parameters
        fitting_tab = self._create_fitting_tab()
        tabs.addTab(fitting_tab, "Fitting")

        # Tab 2: Physical Parameters
        physical_tab = self._create_physical_tab()
        tabs.addTab(physical_tab, "Physical Parameters")

        # Tab 3: Geometry
        geometry_tab = self._create_geometry_tab()
        tabs.addTab(geometry_tab, "Geometry")

        layout.addWidget(tabs)

        # === Buttons ===
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_fitting_tab(self):
        """Create the fitting parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # === Preset Selection ===
        preset_group = QGroupBox("Optimization Preset")
        preset_layout = QFormLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['fast', 'medium', 'slow'])
        self.preset_combo.setCurrentText('medium')
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)

        preset_info = QLabel(
            "fast: ~2 min, 100 sims (10×10 grid)\n"
            "medium: ~5 min, 225 sims (15×15 grid)\n"
            "slow: ~12 min, 625 sims (25×25 grid)"
        )
        preset_info.setStyleSheet(INFO_LABEL_STYLE_SMALL)

        preset_layout.addRow("Preset:", self.preset_combo)
        preset_layout.addRow("", preset_info)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # === Fit Range ===
        fit_range_group = QGroupBox("Fit Range")
        fit_range_layout = QFormLayout()

        self.fit_start_spin = QDoubleSpinBox()
        self.fit_start_spin.setRange(0.01, 100.0)
        self.fit_start_spin.setValue(0.3)
        self.fit_start_spin.setSuffix(" μs")
        self.fit_start_spin.setDecimals(2)

        self.fit_end_spin = QDoubleSpinBox()
        self.fit_end_spin.setRange(1.0, 200.0)
        self.fit_end_spin.setValue(30.0)
        self.fit_end_spin.setSuffix(" μs")
        self.fit_end_spin.setDecimals(1)

        fit_range_layout.addRow("Start time:", self.fit_start_spin)
        fit_range_layout.addRow("End time:", self.fit_end_spin)
        fit_range_group.setLayout(fit_range_layout)
        layout.addWidget(fit_range_group)

        # === Grid Search Parameters ===
        grid_group = QGroupBox("Grid Search Parameters")
        grid_layout = QFormLayout()

        # tau_T range (triplet lifetime)
        self.tau_T_min_spin = QDoubleSpinBox()
        self.tau_T_min_spin.setRange(0.1, 10.0)
        self.tau_T_min_spin.setValue(1.5)  # Typical lower bound
        self.tau_T_min_spin.setSuffix(" μs")
        self.tau_T_min_spin.setDecimals(2)
        self.tau_T_min_spin.setToolTip("Minimum τ_T to search (photosensitizer triplet decay time)")

        self.tau_T_max_spin = QDoubleSpinBox()
        self.tau_T_max_spin.setRange(0.1, 10.0)
        self.tau_T_max_spin.setValue(2.5)  # Typical upper bound
        self.tau_T_max_spin.setSuffix(" μs")
        self.tau_T_max_spin.setDecimals(2)
        self.tau_T_max_spin.setToolTip("Maximum τ_T to search (photosensitizer triplet decay time)")

        tau_T_layout = QHBoxLayout()
        tau_T_layout.addWidget(self.tau_T_min_spin)
        tau_T_layout.addWidget(QLabel("to"))
        tau_T_layout.addWidget(self.tau_T_max_spin)

        # tau_w range (water lifetime)
        self.tau_w_min_spin = QDoubleSpinBox()
        self.tau_w_min_spin.setRange(0.1, 10.0)
        self.tau_w_min_spin.setValue(3.5)  # Typical lower bound
        self.tau_w_min_spin.setSuffix(" μs")
        self.tau_w_min_spin.setDecimals(2)
        self.tau_w_min_spin.setToolTip("Minimum τ_Δ,water to search (¹O₂ decay time in water)")

        self.tau_w_max_spin = QDoubleSpinBox()
        self.tau_w_max_spin.setRange(0.1, 10.0)
        self.tau_w_max_spin.setValue(4.5)  # Typical upper bound
        self.tau_w_max_spin.setSuffix(" μs")
        self.tau_w_max_spin.setDecimals(2)
        self.tau_w_max_spin.setToolTip("Maximum τ_Δ,water to search (¹O₂ decay time in water)")

        tau_w_layout = QHBoxLayout()
        tau_w_layout.addWidget(self.tau_w_min_spin)
        tau_w_layout.addWidget(QLabel("to"))
        tau_w_layout.addWidget(self.tau_w_max_spin)

        # Grid points
        self.grid_points_spin = QSpinBox()
        self.grid_points_spin.setRange(5, 50)  # Increased max for thorough searches
        self.grid_points_spin.setValue(15)
        self.grid_points_spin.setToolTip("Number of grid points per dimension (N×N coarse grid)")

        grid_layout.addRow("τ_T range:", tau_T_layout)
        grid_layout.addRow("τ_w range:", tau_w_layout)
        grid_layout.addRow("Grid points:", self.grid_points_spin)

        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)

        # === Info Label ===
        info_label = QLabel(
            "Grid search will explore τ_T × τ_Δ,water parameter space.\n"
            "Coarse grid: N×N simulations\n"
            "Fine grid: additional ~121 simulations around best fit\n\n"
            "Tip: Start with wider ranges, then refine based on results."
        )
        info_label.setStyleSheet(INFO_LABEL_STYLE_WITH_MARGIN)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()
        return tab

    def _create_physical_tab(self):
        """Create the physical parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        phys_group = QGroupBox("Fixed Physical Parameters")
        phys_layout = QFormLayout()

        # Diffusion coefficients (scientific notation)
        self.D_water_edit = QLineEdit("2.0e-5")
        self.D_water_edit.setToolTip("Diffusion coefficient of ¹O₂ in water (cm²/s)")
        phys_layout.addRow("D_water [cm²/s]:", self.D_water_edit)

        self.D_lipid_edit = QLineEdit("1.0e-5")
        self.D_lipid_edit.setToolTip("Diffusion coefficient of ¹O₂ in lipid (cm²/s)")
        phys_layout.addRow("D_lipid [cm²/s]:", self.D_lipid_edit)

        # tau_delta_lipid (fixed)
        self.tau_delta_lipid_spin = QDoubleSpinBox()
        self.tau_delta_lipid_spin.setRange(1.0, 100.0)
        self.tau_delta_lipid_spin.setValue(14.0)
        self.tau_delta_lipid_spin.setSuffix(" μs")
        self.tau_delta_lipid_spin.setDecimals(1)
        self.tau_delta_lipid_spin.setToolTip("¹O₂ lifetime in DPPC lipid bilayer (fixed)")
        phys_layout.addRow("τ_Δ,lipid:", self.tau_delta_lipid_spin)

        # Partition coefficient
        self.partition_coeff_spin = QDoubleSpinBox()
        self.partition_coeff_spin.setRange(0.1, 100.0)
        self.partition_coeff_spin.setValue(3.5)
        self.partition_coeff_spin.setDecimals(2)
        self.partition_coeff_spin.setToolTip("Partition coefficient S = C_lipid / C_water")
        phys_layout.addRow("Partition coeff (S):", self.partition_coeff_spin)

        phys_group.setLayout(phys_layout)
        layout.addWidget(phys_group)

        info_label = QLabel(
            "These parameters are fixed during fitting and based on\n"
            "Hackbarth & Röder (2015), Photochem. Photobiol. Sci., 14, 329-334"
        )
        info_label.setStyleSheet(INFO_LABEL_STYLE_WITH_MARGIN)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()
        return tab

    def _create_geometry_tab(self):
        """Create the geometry parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        geom_group = QGroupBox("Vesicle Geometry (78 nm SUV)")
        geom_layout = QFormLayout()

        # Number of layers
        self.n_layers_spin = QSpinBox()
        self.n_layers_spin.setRange(50, 1000)
        self.n_layers_spin.setValue(400)
        self.n_layers_spin.setToolTip("Total number of concentric spherical layers")
        geom_layout.addRow("Total layers (N):", self.n_layers_spin)

        # Membrane start
        self.membrane_start_spin = QSpinBox()
        self.membrane_start_spin.setRange(1, 500)
        self.membrane_start_spin.setValue(36)
        self.membrane_start_spin.setToolTip("First layer of lipid bilayer")
        geom_layout.addRow("Membrane start:", self.membrane_start_spin)

        # Membrane end
        self.membrane_end_spin = QSpinBox()
        self.membrane_end_spin.setRange(1, 500)
        self.membrane_end_spin.setValue(39)
        self.membrane_end_spin.setToolTip("Last layer of lipid bilayer (thickness = 4 nm)")
        geom_layout.addRow("Membrane end:", self.membrane_end_spin)

        # PS layers
        self.ps_layers_edit = QLineEdit("37, 38")
        self.ps_layers_edit.setToolTip("Layers where photosensitizer is embedded (comma-separated)")
        geom_layout.addRow("PS layers:", self.ps_layers_edit)

        geom_group.setLayout(geom_layout)
        layout.addWidget(geom_group)

        info_label = QLabel(
            "Layers 1-35: Water inside vesicle\n"
            "Layers 36-39: DPPC lipid bilayer (4 nm)\n"
            "Layers 40-400: Water outside vesicle\n\n"
            "Each layer = 1 nm thickness (fixed)"
        )
        info_label.setStyleSheet(INFO_LABEL_STYLE_WITH_MARGIN)
        layout.addWidget(info_label)

        layout.addStretch()
        return tab

    def _on_preset_changed(self, preset: str):
        """Update grid points based on preset selection."""
        preset_grid_points = {
            'fast': 10,
            'medium': 15,
            'slow': 25
        }
        if preset in preset_grid_points:
            self.grid_points_spin.setValue(preset_grid_points[preset])

    def _validate_and_accept(self):
        """Validate parameters before accepting dialog."""
        from PyQt6.QtWidgets import QMessageBox

        # Validate tau_T range
        if self.tau_T_min_spin.value() >= self.tau_T_max_spin.value():
            QMessageBox.warning(
                self,
                "Invalid Range",
                "τ_T minimum must be less than maximum!\n\n"
                f"Current: {self.tau_T_min_spin.value():.2f} to {self.tau_T_max_spin.value():.2f} μs"
            )
            return

        # Validate tau_w range
        if self.tau_w_min_spin.value() >= self.tau_w_max_spin.value():
            QMessageBox.warning(
                self,
                "Invalid Range",
                "τ_Δ,water minimum must be less than maximum!\n\n"
                f"Current: {self.tau_w_min_spin.value():.2f} to {self.tau_w_max_spin.value():.2f} μs"
            )
            return

        # Validate fit range
        if self.fit_start_spin.value() >= self.fit_end_spin.value():
            QMessageBox.warning(
                self,
                "Invalid Range",
                "Fit start time must be less than end time!\n\n"
                f"Current: {self.fit_start_spin.value():.2f} to {self.fit_end_spin.value():.2f} μs"
            )
            return

        # Validate membrane geometry
        if self.membrane_start_spin.value() >= self.membrane_end_spin.value():
            QMessageBox.warning(
                self,
                "Invalid Geometry",
                "Membrane start layer must be less than end layer!\n\n"
                f"Current: {self.membrane_start_spin.value()} to {self.membrane_end_spin.value()}"
            )
            return

        # Warn if ranges are too narrow
        tau_T_range = self.tau_T_max_spin.value() - self.tau_T_min_spin.value()
        tau_w_range = self.tau_w_max_spin.value() - self.tau_w_min_spin.value()

        if tau_T_range < 0.5:
            response = QMessageBox.question(
                self,
                "Narrow τ_T Range",
                f"τ_T range is very narrow ({tau_T_range:.2f} μs).\n\n"
                "This may miss the true parameter value if your initial guess is off.\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if response == QMessageBox.StandardButton.No:
                return

        if tau_w_range < 0.5:
            response = QMessageBox.question(
                self,
                "Narrow τ_Δ,water Range",
                f"τ_Δ,water range is very narrow ({tau_w_range:.2f} μs).\n\n"
                "This may miss the true parameter value if your initial guess is off.\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if response == QMessageBox.StandardButton.No:
                return

        # Calculate and show estimated time
        grid_points = self.grid_points_spin.value()
        n_coarse = grid_points * grid_points
        n_fine = 11 * 11  # Approximate fine grid size
        n_total = n_coarse + n_fine

        # Estimate: ~1.2s per simulation with Numba
        estimated_minutes = (n_total * 1.2) / 60

        # Show confirmation with estimated time
        response = QMessageBox.information(
            self,
            "Ready to Start",
            f"Grid Search Configuration:\n\n"
            f"τ_T range: {self.tau_T_min_spin.value():.2f} - {self.tau_T_max_spin.value():.2f} μs\n"
            f"τ_Δ,water range: {self.tau_w_min_spin.value():.2f} - {self.tau_w_max_spin.value():.2f} μs\n"
            f"Grid points: {grid_points} × {grid_points}\n\n"
            f"Total simulations: ~{n_total}\n"
            f"Estimated time: ~{estimated_minutes:.1f} minutes\n\n"
            f"Click OK to start analysis.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )

        if response == QMessageBox.StandardButton.Ok:
            self.accept()

    def get_parameters(self):
        """Get ALL configured parameters from all tabs."""
        # Parse PS layers from string "37, 38" to tuple (37, 38)
        ps_layers_str = self.ps_layers_edit.text()
        ps_layers = tuple(int(x.strip()) for x in ps_layers_str.split(','))

        return {
            # Fitting parameters (ACTUALLY USED by fitter now!)
            'preset': self.preset_combo.currentText(),  # For reference only (custom_params override it)
            'fit_start_us': self.fit_start_spin.value(),
            'fit_end_us': self.fit_end_spin.value(),
            'tau_T_min': self.tau_T_min_spin.value(),
            'tau_T_max': self.tau_T_max_spin.value(),
            'tau_w_min': self.tau_w_min_spin.value(),
            'tau_w_max': self.tau_w_max_spin.value(),
            'grid_points': self.grid_points_spin.value(),
            # Physical parameters
            'D_water': float(self.D_water_edit.text()),
            'D_lipid': float(self.D_lipid_edit.text()),
            'tau_delta_lipid': self.tau_delta_lipid_spin.value(),
            'partition_coeff': self.partition_coeff_spin.value(),
            # Geometry parameters
            'n_layers': self.n_layers_spin.value(),
            'membrane_start': self.membrane_start_spin.value(),
            'membrane_end': self.membrane_end_spin.value(),
            'ps_layers': ps_layers
        }
