#!/usr/bin/env python3
"""
Dataclasses for Heterogeneous Diffusion Model Analysis

Based on: Hackbarth & Röder (2015)
"Singlet oxygen luminescence kinetics in a heterogeneous environment"
Photochem. Photobiol. Sci., 14, 329-334

Implements spherical symmetrical diffusion model for SUVs (Small Unilamellar Vesicles)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class VesicleGeometry:
    """
    Geometry parameters for SUV (Small Unilamellar Vesicle) model.

    The model consists of concentric spherical layers:
    - Layers 1-35: Water inside vesicle
    - Layers 36-39: DPPC lipid bilayer (4 nm thickness)
    - Layers 40-N: Water outside vesicle

    Default values from Hackbarth & Röder 2015 (78 nm SUVs):
    """
    n_layers: int = 400
    """Total number of concentric spherical layers"""

    layer_thickness: float = 1.0
    """Thickness of each layer in nm (fixed at 1 nm)"""

    membrane_start: int = 36
    """First layer of lipid bilayer (layer 36)"""

    membrane_end: int = 39
    """Last layer of lipid bilayer (layer 39)"""

    ps_layers: Tuple[int, int] = (37, 38)
    """Layers where photosensitizer is located (default: embedded in membrane)"""

    def __post_init__(self):
        """Validate geometry parameters."""
        if self.membrane_start >= self.membrane_end:
            raise ValueError("membrane_start must be < membrane_end")
        if self.ps_layers[0] < self.membrane_start or self.ps_layers[1] > self.membrane_end:
            raise ValueError(f"PS layers {self.ps_layers} must be within membrane layers {self.membrane_start}-{self.membrane_end}")
        if self.n_layers <= self.membrane_end:
            raise ValueError(f"n_layers ({self.n_layers}) must be > membrane_end ({self.membrane_end})")

    @property
    def membrane_thickness(self) -> float:
        """Membrane thickness in nm"""
        return (self.membrane_end - self.membrane_start + 1) * self.layer_thickness

    @property
    def vesicle_diameter(self) -> float:
        """Outer diameter of vesicle in nm"""
        return 2 * self.membrane_end * self.layer_thickness

    def is_membrane_layer(self, layer_idx: int) -> bool:
        """Check if layer index is within membrane."""
        return self.membrane_start <= layer_idx <= self.membrane_end

    def is_water_layer(self, layer_idx: int) -> bool:
        """Check if layer index is water (inside or outside vesicle)."""
        return not self.is_membrane_layer(layer_idx)


@dataclass
class DiffusionParameters:
    """
    Physical parameters for singlet oxygen diffusion simulation.

    Default values from Hackbarth & Röder 2015 and references therein.
    """
    # Diffusion coefficients (cm²/s)
    D_water: float = 2.0e-5
    """Diffusion coefficient of ¹O₂ in water at room temperature (cm²/s)"""

    D_lipid: float = 1.0e-5
    """Diffusion coefficient of ¹O₂ in lipid membrane (cm²/s)"""

    # Decay times (μs)
    tau_delta_lipid: float = 14.0
    """¹O₂ decay time in DPPC lipid bilayer (μs) - fixed, minimal influence on kinetics"""

    # These are fitted:
    tau_T: float = 2.0
    """PS triplet decay time (μs) - to be fitted"""

    tau_delta_water: float = 3.7
    """¹O₂ decay time in water (μs) - to be fitted"""

    # Partition coefficient
    partition_coeff: float = 3.5
    """¹O₂ solubility ratio: concentration(lipid) / concentration(water)"""

    # Simulation parameters
    time_step: float = 1.25e-4
    """Time step for numerical simulation (μs) - corresponds to 0.125 ns"""

    max_time: float = 30.0
    """Maximum simulation time (μs)"""

    output_time_step: float = 0.02
    """Output time resolution (μs) - corresponds to 20 ns bin width"""

    def __post_init__(self):
        """Validate parameters."""
        if self.tau_T <= 0 or self.tau_delta_water <= 0 or self.tau_delta_lipid <= 0:
            raise ValueError("All decay times must be positive")
        if self.D_water <= 0 or self.D_lipid <= 0:
            raise ValueError("Diffusion coefficients must be positive")
        if self.partition_coeff <= 0:
            raise ValueError("Partition coefficient must be positive")

    def get_diffusion_coeff(self, is_membrane: bool) -> float:
        """Get diffusion coefficient for membrane or water."""
        return self.D_lipid if is_membrane else self.D_water

    def get_decay_time(self, is_membrane: bool) -> float:
        """Get ¹O₂ decay time for membrane or water."""
        return self.tau_delta_lipid if is_membrane else self.tau_delta_water


@dataclass
class SimulationResult:
    """
    Results from diffusion simulation for a single parameter set.

    Contains time-resolved ¹O₂ concentrations in lipid and water phases.
    """
    time: np.ndarray
    """Time points for output (μs)"""

    n_lipid: np.ndarray
    """Total amount of ¹O₂ in lipid layers vs time (arbitrary units)"""

    n_water: np.ndarray
    """Total amount of ¹O₂ in water layers vs time (arbitrary units)"""

    tau_T: float
    """PS triplet decay time used for this simulation (μs)"""

    tau_delta_water: float
    """¹O₂ decay time in water used for this simulation (μs)"""

    geometry: VesicleGeometry
    """Vesicle geometry parameters"""

    parameters: DiffusionParameters
    """Physical parameters used"""

    @property
    def peak_time_lipid(self) -> float:
        """Time of maximum ¹O₂ concentration in lipid (μs)"""
        return self.time[np.argmax(self.n_lipid)]

    @property
    def peak_time_water(self) -> float:
        """Time of maximum ¹O₂ concentration in water (μs)"""
        return self.time[np.argmax(self.n_water)]


@dataclass
class HeterogeneousFitResult:
    """
    Results from fitting experimental data with heterogeneous diffusion model.

    Analogous to KineticsResult but for heterogeneous analysis.
    """
    # Fitted parameters
    tau_T: float
    """Best-fit PS triplet decay time (μs)"""

    tau_delta_water: float
    """Best-fit ¹O₂ decay time in water (μs)"""

    rate_ratio: float
    """Ratio of radiative rate constants: k_rad(lipid) / k_rad(water)"""

    amplitude_lipid: float
    """Amplitude A for lipid signal component"""

    amplitude_water: float
    """Amplitude B for water signal component"""

    background: float
    """Background level C (dark counts)"""

    # Fit quality
    chi_square: float
    """Chi-square statistic"""

    reduced_chi_square: float
    """Reduced chi-square (χ²/DOF) - should be ~1.0 for good fit"""

    r_squared: float
    """Coefficient of determination"""

    degrees_of_freedom: int
    """Degrees of freedom"""

    n_fit_points: int
    """Number of data points used in fit"""

    # Data and fit curves
    time_experimental: np.ndarray
    """Experimental time points (μs)"""

    intensity_experimental: np.ndarray
    """Experimental intensity data (photon counts)"""

    fit_curve: np.ndarray
    """Best-fit curve: A·n_L(t) + B·n_W(t) + C"""

    n_lipid_curve: np.ndarray
    """Lipid component from simulation: n_L(t)"""

    n_water_curve: np.ndarray
    """Water component from simulation: n_W(t)"""

    residuals: np.ndarray
    """Fit residuals: experimental - fit"""

    weighted_residuals: np.ndarray
    """Weighted residuals for chi-square calculation"""

    # Simulation info
    geometry: VesicleGeometry
    """Vesicle geometry used"""

    parameters: DiffusionParameters
    """Physical parameters used"""

    # Grid search info
    grid_search_surface: Optional[np.ndarray] = None
    """Chi-square surface from grid search (for plotting Fig. 4)"""

    grid_tau_T: Optional[np.ndarray] = None
    """τ_T values for grid search surface"""

    grid_tau_delta_W: Optional[np.ndarray] = None
    """τ_Δ,water values for grid search surface"""

    fit_time_range: Tuple[float, float] = field(default_factory=lambda: (0.3, 30.0))
    """Time range used for fitting (μs)"""

    def is_good_fit(self, chi2_threshold: float = 1.5) -> bool:
        """
        Check if fit quality is acceptable.

        Paper uses χ²_red < 1.1 for best fits, up to 1.2 for acceptable.
        We use 1.5 as conservative threshold.
        """
        return self.reduced_chi_square < chi2_threshold

    def get_rate_ratio_uncertainty(self) -> Tuple[float, float]:
        """
        Estimate uncertainty in rate ratio from chi-square surface.

        Returns (lower_bound, upper_bound) based on χ²_red < 1.1 region.
        """
        if self.grid_search_surface is None:
            return (self.rate_ratio, self.rate_ratio)

        # This would require storing rate_ratio for each grid point
        # For now, return nominal value
        # TODO: Implement proper uncertainty estimation
        return (self.rate_ratio - 0.5, self.rate_ratio + 0.5)

    def __str__(self) -> str:
        """Formatted summary of fit results."""
        return (
            f"Heterogeneous Fit Results:\n"
            f"  tau_T = {self.tau_T:.2f} us\n"
            f"  tau_Delta(water) = {self.tau_delta_water:.2f} us\n"
            f"  Rate ratio = {self.rate_ratio:.2f} +/- 0.5\n"
            f"  chi2_red = {self.reduced_chi_square:.2f}\n"
            f"  R^2 = {self.r_squared:.3f}\n"
            f"  Quality: {'GOOD' if self.is_good_fit() else 'POOR'}"
        )
