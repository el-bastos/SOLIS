#!/usr/bin/env python3
"""
Dataclasses for Singlet Oxygen Kinetics Analysis Results

Replaces the 40+ key result dictionary with structured, type-safe dataclasses.
Phase 2 refactoring - clean break from legacy dict-based results.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any
import numpy as np


@dataclass
class SNRResult:
    """
    Signal-to-noise ratio analysis results.

    Calculated using Mode 2 method: SNR = (peak - baseline) / noise_std
    where baseline and noise_std come from the tail baseline region.

    Quality thresholds:
        - Excellent: SNR > 20 dB
        - Good: SNR > 10 dB
        - Fair: SNR > 3 dB
        - Poor: SNR ≤ 3 dB
    """
    snr_db: float
    """Signal-to-noise ratio in decibels (20*log10(linear))"""

    snr_linear: float
    """Signal-to-noise ratio in linear units"""

    quality: str
    """Quality assessment: 'Excellent', 'Good', 'Fair', or 'Poor'"""

    spike_region: Optional[Dict[str, Any]]
    """Spike detection results from TransitionBasedSpikeDetector"""

    beginning_baseline: Dict[str, Any]
    """Beginning baseline statistics (used for spike detection)"""

    tail_baseline: Dict[str, Any]
    """Tail baseline statistics (used for SNR calculation)"""

    baseline_used: str
    """Which baseline was used for SNR: 'tail' or 'beginning'"""

    peak_signal: float = 0.0
    """Peak value in signal region (photon counts)"""

    baseline_level: float = 0.0
    """Baseline level used for SNR calculation (photon counts)"""

    noise_std: float = 0.0
    """Standard deviation of baseline noise (photon counts)"""

    signal_region: Optional[Dict[str, Any]] = None
    """Signal region data: {'x', 'y', 'start_idx'}"""

    def has_spike(self) -> bool:
        """Check if a laser spike was detected."""
        return self.spike_region is not None

    def is_good_quality(self, threshold_db: float = 10.0) -> bool:
        """
        Check if signal quality meets threshold.

        Args:
            threshold_db: Minimum acceptable SNR in dB (default 10.0)

        Returns:
            True if SNR >= threshold
        """
        return self.snr_db >= threshold_db


@dataclass
class FitParameters:
    """
    Fitted parameters from Main model f(t-t0).

    The Main model accounts for signal delay (lag time) and is used as the
    primary result for quantum yield calculations.

    Model:
        Biexponential: A * (τΔ/(τΔ-τT)) * (e^(-(t-t0)/τΔ) - e^(-(t-t0)/τT)) + y0
        Single exponential: A * e^(-(t-t0)/τΔ) + y0
    """
    A: float
    """Amplitude (photon counts) - used for QY calculation"""

    tau_delta: float
    """Singlet oxygen decay time constant (μs)"""

    tau_T: Union[float, str]
    """Triplet state rise time constant (μs), or 'ND' for single exponential"""

    t0: float
    """Lag time / signal delay (μs)"""

    y0: float
    """Baseline offset (photon counts)"""

    def is_single_exponential(self) -> bool:
        """Check if model is single exponential (no triplet rise component)."""
        return self.tau_T == 'ND'


@dataclass
class FitQuality:
    """
    Goodness of fit metrics for TCSPC data.

    TCSPC data follows Poisson statistics (variance = mean), so chi-square
    is the proper metric for quality assessment. Reduced chi-square should
    be approximately 1.0 for good fits.
    """
    r_squared: float
    """Coefficient of determination (0-1, higher is better)"""

    chi_square: float
    """Chi-square statistic: Σ[(observed - predicted)² / intensity]"""

    reduced_chi_square: float
    """Reduced chi-square: χ² / DOF (should be ~1.0 for good Poisson fits)"""

    residual_standard_error: float
    """Root mean square error of residuals"""

    n_fit_points: int
    """Number of data points used in fitting"""

    degrees_of_freedom: int
    """Degrees of freedom: n_fit_points - n_parameters"""

    model_used: str
    """Model type: 'biexponential' or 'single_exponential'"""

    def is_good_fit(self, r2_threshold: float = 0.95,
                    chi2_threshold: float = 2.0) -> bool:
        """
        Check if fit quality meets acceptance criteria.

        Args:
            r2_threshold: Minimum acceptable R² value (default 0.95)
            chi2_threshold: Maximum acceptable reduced χ² (default 2.0)

        Returns:
            True if both criteria are met
        """
        return (self.r_squared >= r2_threshold and
                self.reduced_chi_square <= chi2_threshold)


@dataclass
class LiteratureModelResult:
    """
    Literature model f(t) fitting results.

    This model does not account for signal delay (no t0 parameter).
    Used for comparison purposes only - the Main model is preferred.

    Model:
        Biexponential: A * (τΔ/(τΔ-τT)) * (e^(-t/τΔ) - e^(-t/τT)) + y0
        Single exponential: A * e^(-t/τΔ) + y0
    """
    success: bool
    """Whether the fitting succeeded"""

    A: Optional[float] = None
    """Amplitude parameter (also called S0 in literature)"""

    tau_delta: Optional[float] = None
    """Singlet oxygen decay time (μs)"""

    tau_T: Union[float, str, None] = None
    """Triplet rise time (μs) or 'ND' for single exponential"""

    y0: Optional[float] = None
    """Baseline offset (photon counts)"""

    r_squared: Optional[float] = None
    """Coefficient of determination"""

    chi_square: Optional[float] = None
    """Chi-square statistic"""

    reduced_chi_square: Optional[float] = None
    """Reduced chi-square (χ²/DOF)"""

    model_used: Optional[str] = None
    """'biexponential' or 'single_exponential'"""

    curve: Optional[np.ndarray] = None
    """Full fitted curve on experimental timeline"""

    weighted_residuals: Optional[np.ndarray] = None
    """Poisson-weighted residuals: (observed - predicted) / sqrt(intensity)"""


@dataclass
class WorkflowInfo:
    """
    Metadata about the analysis workflow.

    Records details of the 3-step analysis process for traceability.
    """
    method: str = 'clean_3_step_workflow'
    """Analysis method identifier"""

    baseline_points_removed: int = 0
    """Number of baseline points removed in Step 1"""

    spike_duration: float = 0.0
    """Duration of laser spike artifact (μs)"""

    fitting_points: int = 0
    """Number of points used for fitting (after masking)"""

    step4_status: str = 'deprecated'
    """Status of Step 4 (always 'deprecated' in v2.5+)"""


@dataclass
class KineticsResult:
    """
    Complete kinetics analysis result from 3-step workflow.

    This is the main result object returned by kinetics_analyzer.fit_kinetics().

    Workflow:
        Step 1: Remove baseline (points before spike)
        Step 2: Create spike mask (identify artifact region)
        Step 3: Fit Main and Literature models with chi-square metrics

    The Main model results (parameters, fit_quality) are used for quantum
    yield calculations. The Literature model is for comparison only.
    """

    # === Core Results (PRIMARY - used for QY calculation) ===
    parameters: FitParameters
    """Fitted parameters from Main model f(t-t0)"""

    fit_quality: FitQuality
    """Quality metrics for Main model fit"""

    # === Experimental Timeline Data ===
    time_experiment_us: np.ndarray
    """Time axis (μs) starting at t=0 (spike start)"""

    intensity_raw: np.ndarray
    """Measured photon counts on experimental timeline"""

    # === Main Model Outputs ===
    main_curve: np.ndarray
    """Fitted curve from Main model f(t-t0)"""

    main_weighted_residuals: np.ndarray
    """Poisson-weighted residuals: (observed - predicted) / sqrt(intensity)"""

    # === Literature Model (comparison only) ===
    literature: LiteratureModelResult
    """Literature model f(t) results for comparison"""

    # === Masking Information ===
    fitting_mask: np.ndarray
    """Boolean mask: True=signal region (used for fit), False=spike region"""

    spike_region: np.ndarray
    """Boolean mask: True=spike artifact, False=signal region"""

    # === Workflow Metadata ===
    workflow: WorkflowInfo
    """Analysis workflow metadata"""

    # === Optional: Raw Data (for plotting) ===
    raw_time: Optional[np.ndarray] = None
    """Original time axis before baseline removal"""

    raw_intensity: Optional[np.ndarray] = None
    """Original intensity before baseline removal"""

    # === Phase 3A: SNR Analysis ===
    snr_result: Optional[SNRResult] = None
    """Signal-to-noise ratio analysis results"""

    def __post_init__(self):
        """Validate result integrity after initialization."""
        # Ensure arrays have consistent lengths
        n = len(self.time_experiment_us)
        assert len(self.intensity_raw) == n, "Time and intensity arrays must match"
        assert len(self.main_curve) == n, "Main curve must match timeline length"
        assert len(self.main_weighted_residuals) == n, "Residuals must match timeline length"
        assert len(self.fitting_mask) == n, "Fitting mask must match timeline length"
        assert len(self.spike_region) == n, "Spike region must match timeline length"

        # Validate literature curve if present
        if self.literature.success and self.literature.curve is not None:
            assert len(self.literature.curve) == n, "Literature curve must match timeline length"

    def get_signal_region_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get time and intensity data for signal region only (spike masked out).

        Returns:
            (time_signal, intensity_signal): Arrays containing only signal region data
        """
        return self.time_experiment_us[self.fitting_mask], self.intensity_raw[self.fitting_mask]

    def summary_string(self) -> str:
        """
        Generate a human-readable summary of the fit results.

        Returns:
            Multi-line string with key parameters and quality metrics
        """
        lines = []
        lines.append("=== Kinetics Analysis Results ===")
        lines.append(f"Model: {self.fit_quality.model_used}")
        lines.append(f"A = {self.parameters.A:.1f} counts")
        lines.append(f"tau_delta = {self.parameters.tau_delta:.3f} us")
        if self.parameters.tau_T != 'ND':
            lines.append(f"tau_T = {self.parameters.tau_T:.3f} us")
        lines.append(f"t0 = {self.parameters.t0:.4f} us")
        lines.append(f"y0 = {self.parameters.y0:.1f} counts")
        lines.append(f"R² = {self.fit_quality.r_squared:.4f}")
        lines.append(f"χ²_reduced = {self.fit_quality.reduced_chi_square:.3f}")
        lines.append(f"Fit points: {self.fit_quality.n_fit_points}")
        return "\n".join(lines)


@dataclass
class HeterogeneousFitResult:
    """
    Results from heterogeneous vesicle diffusion model fitting.

    This dataclass stores the complete results from fitting experimental singlet oxygen
    decay data to a physical diffusion model for heterogeneous systems (vesicles, lipid
    bilayers). The fitting uses a grid search over (τT, τw) parameter space, running
    the vesicle simulation model for each point to find the best fit.

    The model simulates singlet oxygen diffusion through discrete spherical shells with
    a lipid bilayer embedded in water, accounting for different diffusion coefficients
    and lifetimes in each phase. The signal is modeled as a linear combination:
        S(t) = A·nL(t) + B·nW(t) + C
    where nL and nW are lipid and water phase populations from simulation.

    Workflow:
        1. Grid search over (τT, τw) space (typically 15×15 = 225 points)
        2. For each point: simulate vesicle diffusion → fit linear readout → compute χ²
        3. Best fit: minimum χ²red across grid
        4. Result contains best-fit parameters, basis curves, and quality metrics
    """

    # === Fitted Lifetime Parameters ===
    tau_T_us: float
    """Triplet state lifetime (μs) - fitted parameter"""

    tau_w_us: float
    """Singlet oxygen lifetime in water phase (μs) - fitted parameter"""

    tau_L_us: float
    """Singlet oxygen lifetime in lipid phase (μs) - fixed parameter"""

    # === Physical Parameters (Fixed) ===
    S: float
    """Partition coefficient (lipid/water concentration ratio) - fixed"""

    Dw_cm2s: float = 2e-5
    """Diffusion coefficient in water (cm²/s) - fixed"""

    Dl_cm2s: float = 1e-5
    """Diffusion coefficient in lipid (cm²/s) - fixed"""

    # === Linear Readout Coefficients ===
    A: float = 0.0
    """Amplitude coefficient for lipid component (fitted via linear regression)"""

    B: float = 0.0
    """Amplitude coefficient for water component (fitted via linear regression)"""

    C: float = 0.0
    """Baseline offset (fitted via linear regression)"""

    rate_ratio: float = 0.0
    """A/B ratio - relative contribution of lipid vs water phases"""

    # === Quality Metrics ===
    chi2_reduced: float = 0.0
    """Reduced chi-square: χ²/(n-3) where n = number of fitted points"""

    num_points_fitted: int = 0
    """Number of experimental data points used in fitting"""

    # === Basis Curves (Simulation Output) ===
    time_basis_us: Optional[np.ndarray] = None
    """Time points for basis curves from simulation (μs)"""

    nL_basis: Optional[np.ndarray] = None
    """Lipid phase population curve nL(t) from simulation"""

    nW_basis: Optional[np.ndarray] = None
    """Water phase population curve nW(t) from simulation"""

    # === Experimental Data and Fit ===
    time_exp_us: Optional[np.ndarray] = None
    """Experimental time points (μs)"""

    signal_exp: Optional[np.ndarray] = None
    """Experimental signal (photon counts)"""

    signal_fit: Optional[np.ndarray] = None
    """Fitted signal: A·nL + B·nW + C (photon counts)"""

    residuals: Optional[np.ndarray] = None
    """Residuals: signal_exp - signal_fit"""

    weighted_residuals: Optional[np.ndarray] = None
    """Poisson-weighted residuals: (signal_exp - signal_fit) / sqrt(signal_exp)"""

    # === Grid Search Results ===
    grid_chi2: Optional[Any] = None
    """DataFrame with grid search results: columns = [tau_T_us, tau_w_us, chi2_red, A, B, C]"""

    # === Fitting Metadata ===
    method: str = "Grid Search"
    """Fitting method used (always 'Grid Search' for production)"""

    fit_window: tuple = (0.0, 30.0)
    """Time window used for fitting (start_us, end_us)"""

    system_params: Optional[Dict[str, Any]] = None
    """Complete system parameters dictionary (geometry, physical constants, etc.)"""

    compound_name: str = ""
    """Compound/sample name"""

    replicate_id: str = ""
    """Replicate identifier (e.g., 'Rep1', 'Rep2')"""

    # === Convergence Information (for optimization methods) ===
    num_iterations: Optional[int] = None
    """Number of iterations (for optimization methods, not grid search)"""

    convergence_history: Optional[list] = None
    """Chi-square values at each iteration (for optimization methods)"""

    def get_lipid_component(self) -> Optional[np.ndarray]:
        """
        Get the lipid component curve: A × nL(t).

        Returns:
            Lipid component array, or None if basis curves not available
        """
        if self.nL_basis is not None and self.A is not None:
            return self.A * self.nL_basis
        return None

    def get_water_component(self) -> Optional[np.ndarray]:
        """
        Get the water component curve: B × nW(t).

        Returns:
            Water component array, or None if basis curves not available
        """
        if self.nW_basis is not None and self.B is not None:
            return self.B * self.nW_basis
        return None

    def is_good_fit(self, chi2_threshold: float = 2.0) -> bool:
        """
        Check if fit quality meets acceptance criteria.

        Args:
            chi2_threshold: Maximum acceptable reduced χ² (default 2.0)

        Returns:
            True if χ²red <= threshold
        """
        return self.chi2_reduced <= chi2_threshold

    def summary_string(self) -> str:
        """
        Generate a human-readable summary of the fit results.

        Returns:
            Multi-line string with key parameters and quality metrics
        """
        lines = []
        lines.append("=== Heterogeneous Fit Results ===")
        lines.append(f"Compound: {self.compound_name} ({self.replicate_id})")
        lines.append("")
        lines.append("Fitted Parameters:")
        lines.append(f"  τT = {self.tau_T_us:.3f} μs (triplet lifetime)")
        lines.append(f"  τw = {self.tau_w_us:.3f} μs (singlet O2 in water)")
        lines.append("")
        lines.append("Fixed Parameters:")
        lines.append(f"  τL = {self.tau_L_us:.3f} μs (singlet O2 in lipid)")
        lines.append(f"  S = {self.S:.2f} (partition coefficient)")
        lines.append("")
        lines.append("Linear Readout Coefficients:")
        lines.append(f"  A = {self.A:.4f} (lipid)")
        lines.append(f"  B = {self.B:.4f} (water)")
        lines.append(f"  C = {self.C:.4f} (baseline)")
        lines.append(f"  A/B ratio = {self.rate_ratio:.3f}")
        lines.append("")
        lines.append("Quality Metrics:")
        lines.append(f"  χ²red = {self.chi2_reduced:.3f}")
        lines.append(f"  Fitted points: {self.num_points_fitted}")
        lines.append(f"  Fit window: {self.fit_window[0]:.1f} - {self.fit_window[1]:.1f} μs")
        return "\n".join(lines)

    def get_interpretation(self) -> str:
        """
        Generate interpretation of results based on A/B ratio.

        Returns:
            Human-readable interpretation string
        """
        if self.rate_ratio > 2.0:
            interpretation = "Predominantly lipid environment (A/B > 2)"
        elif self.rate_ratio > 1.0:
            interpretation = "Lipid-dominant mixed environment (1 < A/B < 2)"
        elif self.rate_ratio > 0.5:
            interpretation = "Balanced lipid/water environment (0.5 < A/B < 1)"
        else:
            interpretation = "Water-dominant environment (A/B < 0.5)"

        return interpretation

    def get_quality_assessment(self) -> str:
        """
        Assess fit quality based on χ²red value.

        Returns:
            Quality assessment string
        """
        if self.chi2_reduced <= 1.2:
            return "Excellent fit (χ²red ≤ 1.2)"
        elif self.chi2_reduced <= 2.0:
            return "Good fit (χ²red ≤ 2.0)"
        elif self.chi2_reduced <= 5.0:
            return "Acceptable fit (χ²red ≤ 5.0)"
        else:
            return "Poor fit (χ²red > 5.0) - consider adjusting parameters"


# === Convenience function for backward compatibility ===

def result_to_dict(result: KineticsResult) -> dict:
    """
    Convert KineticsResult to legacy dictionary format.

    Used temporarily during migration to support legacy code.
    This function will be deprecated once all modules are updated.

    Args:
        result: KineticsResult instance

    Returns:
        Dictionary with legacy field names
    """
    return {
        # Direct parameters (for statistical_analyzer)
        'A': result.parameters.A,
        'tau_delta': result.parameters.tau_delta,
        'tau_T': result.parameters.tau_T,
        't0': result.parameters.t0,
        'y0': result.parameters.y0,
        'r_squared': result.fit_quality.r_squared,
        'chi_square': result.fit_quality.chi_square,
        'reduced_chi_square': result.fit_quality.reduced_chi_square,
        'model_used': result.fit_quality.model_used,

        # Experimental timeline data
        'time_experiment_us': result.time_experiment_us,
        'intensity_raw': result.intensity_raw,
        'main_curve_ft_t0': result.main_curve,
        'main_weighted_residuals': result.main_weighted_residuals,
        'literature_curve_ft': result.literature.curve,
        'literature_weighted_residuals': result.literature.weighted_residuals,
        'fitting_mask': result.fitting_mask.astype(int),
        'spike_region': result.spike_region.astype(int),

        # Raw data
        'raw_time': result.raw_time,
        'raw_intensity': result.raw_intensity,

        # Legacy fields for compatibility
        'residual_standard_error': result.fit_quality.residual_standard_error,
        'n_fit_points': result.fit_quality.n_fit_points,

        # Literature model as nested dict
        'literature_refit': {
            'success': result.literature.success,
            'A': result.literature.A,
            'tau_delta': result.literature.tau_delta,
            'tau_T': result.literature.tau_T,
            'y0': result.literature.y0,
            'r_squared': result.literature.r_squared,
            'chi_square': result.literature.chi_square,
            'reduced_chi_square': result.literature.reduced_chi_square,
            'model_used': result.literature.model_used
        },

        # Workflow info
        'workflow_info': {
            'method': result.workflow.method,
            'baseline_points_removed': result.workflow.baseline_points_removed,
            'spike_duration': result.workflow.spike_duration,
            'fitting_points': result.workflow.fitting_points,
            'step4_status': result.workflow.step4_status
        }
    }
