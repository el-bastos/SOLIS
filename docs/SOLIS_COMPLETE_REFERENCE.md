# SOLIS Complete Reference
## Comprehensive Module Documentation

**Last Updated:** 2025-11-02
**Version:** 1.0 Beta
**Purpose:** Complete technical reference for all SOLIS modules

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Module Dependency Graph](#module-dependency-graph)
4. [Core Modules](#core-modules)
5. [GUI Components](#gui-components)
6. [Data Layer](#data-layer)
7. [Plotting System](#plotting-system)
8. [Heterogeneous Analysis](#heterogeneous-analysis)
9. [Surplus Analysis](#surplus-analysis)
10. [Utils & Infrastructure](#utils--infrastructure)
11. [Entry Points](#entry-points)
12. [Test Files](#test-files)
13. [Files to Archive](#files-to-archive)
14. [Critical Usage Patterns](#critical-usage-patterns)

---

## Project Overview

**SOLIS** (Singlet Oxygen Luminescence Investigation System) is a PyQt6-based GUI application for analyzing singlet oxygen (¹O₂) decay kinetics from time-resolved luminescence measurements.

**Key Features:**
- Homogeneous system analysis (biexponential/single exponential fitting)
- Heterogeneous system analysis (vesicle/membrane diffusion models)
- Surplus method for heterogeneous systems
- Quantum yield calculations
- SNR analysis with spike detection
- Linearity studies (concentration, excitation energy)
- Session save/load functionality
- Publication-quality plotting

**Technology Stack:**
- Python 3.10+
- PyQt6 (GUI framework)
- NumPy, SciPy (numerical analysis)
- Matplotlib, Plotly (visualization)
- Numba (JIT compilation for diffusion simulations)
- Pandas (data handling)

**Lines of Code:** ~14,820 (across 45 files)

---

## Directory Structure

```
SOLIS_CLEAN/
├── show_splash_then_load.py          # Main entry point with splash screen
├── solis_gui.py                      # Main GUI window
├── analyze_data_parameters.py        # ⚠️ TEST - Data analysis debug script
├── test_*.py (5 files)               # ⚠️ TEST - Testing scripts
│
├── core/                             # ✅ Backend analysis engine (9 files)
│   ├── __init__.py
│   ├── kinetics_dataclasses.py       # Data structures (KineticsResult, etc.)
│   ├── core_fitting.py               # Low-level fitting functions
│   ├── masking_methods.py            # Spike masking logic
│   ├── spike_detector_last.py        # Advanced spike detection (1108 lines)
│   ├── snr_analyzer.py               # SNR calculation
│   ├── kinetics_analyzer.py          # Main fitting engine (470 lines)
│   ├── statistical_analyzer.py       # Cross-replicate statistics
│   └── quantum_yield_calculator.py   # QY calculations
│
├── gui/                              # ✅ User interface (9 files)
│   ├── __init__.py
│   ├── integrated_browser_widget.py  # File browser + data selector (1374 lines)
│   ├── analysis_worker.py            # Background analysis thread (374 lines)
│   ├── splash_screen.py              # Animated loading screen
│   ├── plot_viewer_widget.py         # Plot display widget (1176 lines)
│   ├── preferences_dialog.py         # Settings dialog
│   ├── dataset_classification_dialog.py  # Dataset type classifier
│   ├── heterogeneous_dialog.py       # Heterogeneous analysis UI (562 lines)
│   └── variable_study_widget.py      # Linearity check UI (705 lines)
│
├── data/                             # ✅ Data parsing (2 files)
│   ├── __init__.py
│   └── file_parser.py                # CSV parser with validation (596 lines)
│
├── plotting/                         # ✅ Visualization (3 files)
│   ├── __init__.py
│   ├── solis_plotter.py              # Main plotter (962 lines)
│   └── variable_study_plotter.py     # Linearity plots (706 lines)
│
├── heterogeneous/                    # ✅ Vesicle/membrane analysis (8 files)
│   ├── __init__.py
│   ├── heterogeneous_dataclasses.py  # Data structures
│   ├── diffusion_simulator_numba.py  # Numba-accelerated diffusion (430 lines)
│   ├── grid_search.py                # Parameter optimization (418 lines)
│   ├── heterogeneous_fitter.py       # Fitting engine (364 lines)
│   ├── heterogeneous_plotter_new.py  # Plotting
│   ├── grid_search_BACKUP_twostep.py         # 🗑️ BACKUP - Archive
│   └── heterogeneous_fitter_BACKUP_twostep.py # 🗑️ BACKUP - Archive
│
├── surplus/                          # ✅ Surplus method (2 files)
│   ├── __init__.py
│   └── surplus_analyzer.py           # 4-step surplus analysis (416 lines)
│
├── utils/                            # ✅ Infrastructure (4 files)
│   ├── __init__.py
│   ├── logger_config.py              # Centralized logging
│   ├── session_manager.py            # Session save/load (534 lines)
│   ├── csv_exporter.py               # Results export
│   └── SOLIS_logo.jpg                # Application logo (800x271 px)
│
├── docs/                             # Documentation
│   ├── BACKEND_REFERENCE.md          # Backend module reference
│   ├── MEMORY.md                     # Development history (66k tokens)
│   └── SOLIS_COMPLETE_REFERENCE.md   # This file
│
└── examples/                         # Example datasets
    ├── homogeneous/
    ├── heterogeneous/
    └── linearity_check/
```

**File Classification:**
- ✅ **Active (37 files):** Core application code
- ⚠️ **Test (6 files):** Should be moved to `/test/`
- 🗑️ **Backup (2 files):** Should be moved to `/old/`

---

## Module Dependency Graph

### Level 0 (No dependencies)
```
└── utils/
    ├── logger_config.py              # Logging configuration
    └── kinetics_dataclasses.py       # (in core/) Data structures
```

### Level 1 (Depends on Level 0)
```
├── core/
│   ├── core_fitting.py               # → numpy, scipy
│   └── spike_detector_last.py        # → numpy
└── heterogeneous/
    └── heterogeneous_dataclasses.py  # → dataclasses, numpy
```

### Level 2 (Depends on Level 0-1)
```
├── core/
│   ├── masking_methods.py            # → core_fitting, spike_detector_last
│   └── snr_analyzer.py               # → spike_detector_last, kinetics_dataclasses
└── data/
    └── file_parser.py                # → pandas, numpy, pathlib
```

### Level 3 (Depends on Level 0-2)
```
├── core/
│   ├── kinetics_analyzer.py          # → masking_methods, snr_analyzer
│   ├── statistical_analyzer.py       # → kinetics_dataclasses
│   └── quantum_yield_calculator.py   # → kinetics_dataclasses
├── heterogeneous/
│   ├── diffusion_simulator_numba.py  # → heterogeneous_dataclasses, numba
│   └── grid_search.py                # → diffusion_simulator_numba
└── surplus/
    └── surplus_analyzer.py           # → kinetics_analyzer
```

### Level 4 (Depends on Level 0-3)
```
├── plotting/
│   ├── solis_plotter.py              # → kinetics_dataclasses, matplotlib
│   └── variable_study_plotter.py     # → plotly, sklearn
├── heterogeneous/
│   ├── heterogeneous_fitter.py       # → grid_search, diffusion_simulator_numba
│   └── heterogeneous_plotter_new.py  # → heterogeneous_dataclasses
└── utils/
    ├── session_manager.py            # → all dataclasses, json
    └── csv_exporter.py               # → kinetics_dataclasses, pandas
```

### Level 5 (GUI layer - Depends on all backend)
```
└── gui/
    ├── analysis_worker.py            # → kinetics_analyzer, statistical_analyzer, qy_calculator
    ├── integrated_browser_widget.py  # → file_parser, dataset_classification_dialog
    ├── plot_viewer_widget.py         # → solis_plotter
    ├── heterogeneous_dialog.py       # → heterogeneous_fitter, heterogeneous_plotter
    ├── variable_study_widget.py      # → variable_study_plotter
    ├── preferences_dialog.py         # → PyQt6
    ├── dataset_classification_dialog.py  # → PyQt6
    └── splash_screen.py              # → PyQt6
```

### Level 6 (Application entry)
```
solis_gui.py                          # → All GUI modules, session_manager
    ↑
show_splash_then_load.py              # → solis_gui, splash_screen (MAIN ENTRY)
```

---

## Core Modules

### 1. kinetics_dataclasses.py
**Location:** `core/kinetics_dataclasses.py` (285 lines)
**Purpose:** Type-safe data structures for analysis results
**Status:** ✅ ACTIVE - Critical core module

**Dependencies:**
```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
```

**Exports:**

#### `@dataclass FitParameters`
Fitted kinetic parameters from decay analysis.
```python
A: float                    # Amplitude (for QY calculation)
tau_delta: float            # Singlet oxygen lifetime (μs)
tau_T: Optional[float]      # Triplet rise time (μs) or None
t0: float                   # Lag time / time shift (μs)
y0: float                   # Baseline offset
errors: Dict[str, float]    # Parameter uncertainties
```

#### `@dataclass FitQuality`
Goodness-of-fit metrics.
```python
r_squared: float            # R² (0-1, closer to 1 is better)
chi_square: float           # χ² statistic
reduced_chi_square: float   # χ²ᵣ (should be ~1.0)
model_used: str             # 'biexponential' or 'single_exponential'
aic: Optional[float]        # Akaike Information Criterion
bic: Optional[float]        # Bayesian Information Criterion
```

#### `@dataclass MaskInfo`
Spike masking information.
```python
spike_start_us: float       # Spike start time (μs)
spike_end_us: float         # Spike end time (μs)
mask_type: str              # 'auto' or 'manual'
points_masked: int          # Number of masked points
```

#### `@dataclass SNRResult`
Signal-to-noise ratio analysis results.
```python
snr_db: float               # SNR in decibels (for internal use)
snr_linear: float           # SNR as ratio (ALWAYS DISPLAY THIS!)
quality: str                # 'Excellent', 'Good', 'Fair', 'Poor'
peak_signal: float          # Peak intensity
noise_std: float            # Noise standard deviation
baseline_mean: float        # Baseline mean
spike_region: Optional[Dict]  # Spike detection info

# Methods
def has_spike() -> bool     # Check if spike detected
def is_good_quality(threshold: float) -> bool  # Check SNR quality
```

#### `@dataclass KineticsResult`
Complete kinetics analysis result container.
```python
# Core results
parameters: FitParameters
fit_quality: FitQuality
snr_result: SNRResult
mask_info: MaskInfo

# Data arrays
time_experiment_us: np.ndarray     # Time axis (μs)
intensity_raw: np.ndarray          # Raw intensity data
intensity_fitted: np.ndarray       # Fitted curve (main model)
residuals: np.ndarray              # Residuals (raw - fitted)
weighted_residuals: np.ndarray     # Weighted residuals

# Metadata
compound: str
replicate_index: int
wavelength: float              # Excitation wavelength (nm)
tau_delta_fixed: Optional[float]  # If tau_delta was fixed

# Methods
def get_signal_region_data() -> Tuple[np.ndarray, np.ndarray]
    """Returns (time, intensity) for unmasked signal region only."""

def summary_string() -> str
    """Returns human-readable summary of fit results."""
```

#### `@dataclass HeterogeneousFitResult`
Results from heterogeneous diffusion fitting.
```python
# Best fit parameters
tau_T: float                # Triplet decay time (μs)
tau_delta_W: float          # Singlet O2 lifetime in water (μs)
tau_delta_L: float          # Singlet O2 lifetime in lipid (μs)

# Fit quality
r_squared: float
chi_square: float
reduced_chi_square: float

# Data
time_us: np.ndarray
intensity_exp: np.ndarray
intensity_fitted: np.ndarray
residuals: np.ndarray
weighted_residuals: np.ndarray

# Grid search info
grid_search_info: Optional[Dict]  # Parameter space explored
```

**Key Notes:**
- ❌ NEVER use dictionaries: `result['A']` ← WRONG!
- ✅ ALWAYS use dataclass attributes: `result.parameters.A` ← CORRECT!
- All arrays are NumPy arrays
- Immutable by design (use `replace()` to modify)

**Used By:** ALL analysis modules, GUI, plotting, export

---

### 2. logger_config.py
**Location:** `utils/logger_config.py` (81 lines)
**Purpose:** Centralized logging configuration
**Status:** ✅ ACTIVE - Core infrastructure

**Dependencies:**
```python
import logging
import sys
from pathlib import Path
```

**Exports:**
```python
def setup_logger(name: str, level=logging.INFO, log_file: str = None) -> logging.Logger
    """Create and configure a logger with specified name and level."""

def get_logger(name: str) -> logging.Logger
    """Get or create default logger for a module."""

def set_global_log_level(level: int):
    """Set log level for all existing loggers."""
```

**Usage Pattern:**
```python
from utils.logger_config import get_logger

logger = get_logger(__name__)  # __name__ = module name

logger.debug("Detailed debug info")
logger.info("Analysis complete")
logger.warning("Low SNR detected: 3.2:1")
logger.error("Fit failed: negative tau_T")
logger.critical("Data file corrupted")
```

**Log Format:**
```
2025-11-02 14:23:45 [INFO] core.kinetics_analyzer: Fit converged (R²=0.998)
```

**Key Notes:**
- ❌ NEVER use `print()` for status messages
- ✅ ALWAYS use logger
- Logs to both console and optional file
- Thread-safe

**Used By:** ALL modules (imported everywhere)

---

### 3. core_fitting.py
**Location:** `core/core_fitting.py` (130 lines)
**Purpose:** Low-level fitting functions and parameter handling
**Status:** ✅ ACTIVE - Core utility

**Dependencies:**
```python
import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional
```

**Exports:**

#### Function: `r2_score(y_true, y_pred) -> float`
Robust R² calculation with numerical stability checks.
```python
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² with checks for:
    - Zero variance in y_true
    - NaN/Inf values
    - Array length mismatch

    Returns: R² value (0-1), or 0.0 if calculation fails
    """
```

#### Class: `CoreFittingMethods`
Base class providing fundamental fitting operations.

**Attributes:**
```python
tau_delta_default: float = 3.5  # Default singlet O2 lifetime
fitted_t0: Optional[float] = None  # Last fitted lag time
```

**Key Methods:**
```python
@staticmethod
def biexponential_model(t, A, tau_delta, tau_T, t0, y0):
    """
    Biexponential model: f(t-t0) = A * (exp(-(t-t0)/tau_delta) - exp(-(t-t0)/tau_T)) + y0
    Clipped at t < t0 → returns y0
    """

@staticmethod
def single_exponential_model(t, A, tau_delta, t0, y0):
    """Single exponential: f(t-t0) = A * exp(-(t-t0)/tau_delta) + y0"""

def detect_and_correct_parameter_exchange(A, tau_delta, tau_T, y0, expected_tau_delta=3.5):
    """
    Detect if curve_fit swapped tau_delta ↔ tau_T.
    Common issue: fitter assigns shorter lifetime to tau_delta.

    Returns: (A_corrected, tau_delta_corrected, tau_T_corrected, y0)
    """

def calculate_weighted_residuals(residuals, intensities):
    """
    Poisson weighting: w_res = residual / sqrt(intensity)
    For low intensity: w_res = residual / sqrt(max(intensity, 1.0))
    """

def calculate_auc(A, tau_delta, tau_T, y0):
    """
    Area under curve (integral from 0 to infinity).
    For biexponential: AUC = A * (tau_delta - tau_T)
    """
```

**Key Algorithms:**
- **Parameter exchange detection:** Checks if tau_delta > tau_T after fit (should be opposite)
- **Poisson weighting:** For counting statistics (photon detection)

**Used By:** masking_methods, kinetics_analyzer

---

### 4. spike_detector_last.py
**Location:** `core/spike_detector_last.py` (1108 lines)
**Purpose:** Advanced transition-based spike detection with adaptive thresholds
**Status:** ✅ ACTIVE - Critical detector (extensively commented)

**Dependencies:**
```python
import numpy as np
from typing import Tuple, Optional, Dict
```

**Exports:**

#### Class: `TransitionBasedSpikeDetector`
State-of-the-art spike detector using transition analysis.

**Key Concept:**
Real signal decays smoothly (negative slope), while spike ends abruptly (positive transition). Detector finds this transition point adaptively.

**Initialization:**
```python
detector = TransitionBasedSpikeDetector(
    transition_threshold_factor=2.0,  # Sensitivity (lower = more sensitive)
    min_spike_duration_us=0.1,        # Minimum spike duration
    baseline_percentile=10.0          # Baseline estimation percentile
)
```

**Main Method:**
```python
def detect_spikes(
    x_data: np.ndarray,           # Time (μs)
    y_data: np.ndarray,           # Intensity
    dataset_type: str = 'auto'    # 'lag_spike', 'spike_only', 'clean_signal', 'preprocessed'
) -> Dict[str, Any]:
    """
    Detect spike region and baselines.

    Returns dict with keys:
    - 'spike_start': Start index
    - 'spike_end': End index
    - 'baseline_initial': Mean of first 10 points
    - 'baseline_final': Mean of last 20% of data (better SNR!)
    - 'has_spike': bool
    - 'confidence': float (0-1)
    """
```

**Dataset Types:**
- `'lag_spike'`: Lag region before spike, then spike, then signal
- `'spike_only'`: Spike starts immediately at t=0
- `'clean_signal'`: No spike (pure decay, e.g., CW excitation)
- `'preprocessed'`: Data already cleaned (trust as-is)

**Advanced Methods:**
```python
def find_baseline_region_at_end(y_data: np.ndarray, percentile: float = 20.0):
    """
    Find baseline from TAIL (last 20%) instead of beginning.
    WHY: Tail has more points → better statistics, lower noise in baseline estimate.
    """

def _find_spike_end_by_transition_analysis(x_data, y_data, spike_start_idx):
    """
    Core algorithm:
    1. Compute first derivative (slope)
    2. Find positive transitions (slope goes from negative to positive)
    3. Score transitions by magnitude
    4. Select strongest transition as spike end
    """

def _validate_spike_region(spike_start_idx, spike_end_idx, y_data):
    """
    Sanity checks:
    - Spike end > spike start
    - Spike duration reasonable
    - Intensity drops after spike
    """
```

**Key Features:**
- **Adaptive thresholding:** Adjusts to data characteristics
- **Robust to noise:** Uses smoothing and percentile-based stats
- **Dataset-aware:** Different strategies for different data types
- **Tail baseline:** Better SNR for baseline estimation

**Algorithm Flow:**
```
1. Determine dataset type (auto or user-specified)
2. Find spike start (first significant rise)
3. Find spike end (transition analysis)
4. Validate spike region
5. Estimate baselines (initial + final)
6. Return detection dict
```

**Used By:** snr_analyzer, masking_methods

**Code Quality:** 🌟 Excellent
- 1108 lines with extensive comments
- Docstrings for every method
- Clear variable names
- Edge case handling

---

### 5. snr_analyzer.py
**Location:** `core/snr_analyzer.py` (136 lines)
**Purpose:** Signal-to-noise ratio calculation
**Status:** ✅ ACTIVE - Core analyzer

**Dependencies:**
```python
import numpy as np
from core.spike_detector_last import TransitionBasedSpikeDetector
from core.kinetics_dataclasses import SNRResult
from utils.logger_config import get_logger
```

**Exports:**

#### Class: `SNRAnalyzer`

**Initialization:**
```python
analyzer = SNRAnalyzer(
    spike_detector=None,  # Uses TransitionBasedSpikeDetector if None
    dataset_type='auto'   # Or 'lag_spike', 'spike_only', etc.
)
```

**Main Method:**
```python
def analyze_snr(
    x_data: np.ndarray,     # Time (μs)
    y_data: np.ndarray,     # Intensity
    dataset_type: str = None  # Override instance default
) -> SNRResult:
    """
    Calculate SNR with spike detection.

    Algorithm:
    1. Detect spike region using spike_detector
    2. Extract signal region (after spike)
    3. Calculate peak_signal = max(signal_region)
    4. Calculate noise_std from baseline_final (tail, not beginning!)
    5. SNR_linear = (peak_signal - baseline_mean) / noise_std
    6. SNR_dB = 20 * log10(SNR_linear)
    7. Classify quality: Excellent (>20dB), Good (>10dB), Fair (>3dB), Poor (≤3dB)

    Returns: SNRResult dataclass
    """
```

**SNR Calculation:**
```python
SNR_linear = (peak_intensity - baseline_mean) / noise_std
SNR_dB = 20 * log10(SNR_linear)
```

**Quality Classification:**
```python
if snr_db >= 20.0:
    quality = 'Excellent'  # SNR > 100:1
elif snr_db >= 10.0:
    quality = 'Good'       # SNR > 10:1
elif snr_db >= 3.0:
    quality = 'Fair'       # SNR > 2:1
else:
    quality = 'Poor'       # SNR ≤ 2:1
```

**Display Convention:**
```python
# ❌ NEVER show dB to users (confusing!)
print(f"SNR: {snr_result.snr_db:.1f} dB")  # WRONG!

# ✅ ALWAYS show linear ratio (clear, intuitive)
print(f"SNR: {snr_result.snr_linear:.1f}:1")  # CORRECT! "52.6:1"
```

**Key Features:**
- Uses tail baseline (better statistics than initial baseline)
- Robust to spike artifacts
- Quality classification for user guidance
- Handles edge cases (zero noise, all NaN, etc.)

**Used By:** gui.analysis_worker, core.kinetics_analyzer

---

### 6. masking_methods.py
**Location:** `core/masking_methods.py` (398 lines)
**Purpose:** Spike masking with auto/manual modes
**Status:** ✅ ACTIVE - Critical masking engine

**Dependencies:**
```python
import numpy as np
from typing import Tuple, Optional
from core.core_fitting import CoreFittingMethods
from core.spike_detector_last import TransitionBasedSpikeDetector
```

**Exports:**

#### Class: `MaskingMethods` (extends `CoreFittingMethods`)

**Key Concept:**
Mask (exclude) spike region from fitting to avoid artifact contamination. Uses boolean mask: `True` = fit this point, `False` = ignore (spike).

**Initialization:**
```python
masker = MaskingMethods(
    spike_detector=None,  # Uses TransitionBasedSpikeDetector if None
    dataset_type='auto'
)
```

**Main Methods:**

##### Auto-masking
```python
def auto_mask_spike(
    time_us: np.ndarray,
    intensity: np.ndarray,
    dataset_type: str = 'auto'
) -> Tuple[np.ndarray, Dict]:
    """
    Automatically detect and mask spike.

    Returns:
    - mask: boolean array (True = fit, False = ignore)
    - mask_info: Dict with 'spike_start_us', 'spike_end_us', 'mask_type'
    """
```

##### Manual masking
```python
def apply_custom_mask(
    time_us: np.ndarray,
    spike_end_us: float
) -> Tuple[np.ndarray, Dict]:
    """
    Create mask with user-specified spike end time.
    Mask all points with time < spike_end_us.

    Returns:
    - mask: boolean array
    - mask_info: Dict with mask_type='manual'
    """
```

##### Time shifting (internal)
```python
def _shift_time_to_spike_start(time_us, intensity, spike_start_us):
    """
    Shift time axis so spike_start → t=0.
    WHY: Fit model expects t=0 at pulse arrival.

    Returns: (time_shifted, intensity_shifted)
    """
```

**Masking Strategy:**
```
Original data:    [----------------spike-------|--------signal-------]
                  0       spike_start    spike_end              max_t

Mask array:       [False False ... False False | True True ... True  ]
                                                 ↑
                                            Fit starts here

Time shift:       [--------spike-------|--------signal-------]
                  -t_shift            0                   max_t
                                      ↑
                                  New t=0
```

**Key Features:**
- **Auto mode:** Uses spike_detector for hands-off operation
- **Manual mode:** User overrides spike end (via GUI preview plots)
- **Time shifting:** Aligns model t=0 with physical pulse arrival
- **Validation:** Checks mask integrity before fitting

**Used By:** core.kinetics_analyzer

---

### 7. kinetics_analyzer.py
**Location:** `core/kinetics_analyzer.py` (470 lines)
**Purpose:** Main kinetics fitting engine with 3-step workflow
**Status:** ✅ ACTIVE - Core analysis engine

**Dependencies:**
```python
import numpy as np
from scipy.optimize import curve_fit
from core.masking_methods import MaskingMethods
from core.kinetics_dataclasses import (
    KineticsResult, FitParameters, FitQuality, MaskInfo, SNRResult
)
from core.snr_analyzer import SNRAnalyzer
from utils.logger_config import get_logger
```

**Exports:**

#### Class: `KineticsAnalyzer` (extends `MaskingMethods`)

The heart of SOLIS homogeneous analysis.

**Initialization:**
```python
analyzer = KineticsAnalyzer(
    tau_delta_default=3.5,      # Default singlet O2 lifetime
    spike_detector=None,        # Auto-creates if None
    dataset_type='auto'         # Or 'lag_spike', 'spike_only', etc.
)
```

**Main Method:**
```python
def fit_kinetics(
    time_us: np.ndarray,
    intensity: np.ndarray,
    tau_delta_fixed: Optional[float] = None,  # Fix tau_delta for standard
    dataset_type: str = None,                 # Override instance default
    custom_mask_end_us: Optional[float] = None  # Manual mask override
) -> KineticsResult:
    """
    Complete kinetics analysis pipeline.

    3-Step Workflow:
    1. Mask spike region (auto or manual)
    2. Calculate SNR
    3. Fit model (biexponential or single exponential)

    Returns: KineticsResult with all data + fit results
    """
```

**3-Step Workflow:**

##### Step 1: Spike Masking
```python
if custom_mask_end_us is not None:
    mask, mask_info = self.apply_custom_mask(time_us, custom_mask_end_us)
else:
    mask, mask_info = self.auto_mask_spike(time_us, intensity, dataset_type)
```

##### Step 2: SNR Calculation
```python
snr_analyzer = SNRAnalyzer(self.spike_detector, dataset_type)
snr_result = snr_analyzer.analyze_snr(time_us, intensity, dataset_type)
```

##### Step 3: Model Fitting
```python
# Try biexponential first
try:
    fit_result = self._fit_biexponential(
        time_us[mask],
        intensity[mask],
        tau_delta_fixed
    )
except:
    # Fallback to single exponential
    fit_result = self._fit_single_exponential(
        time_us[mask],
        intensity[mask],
        tau_delta_fixed
    )
```

**Model Selection Logic:**

Biexponential is preferred, but switches to single exponential if:
1. `tau_T < 0.05 μs` (too fast, unphysical)
2. `|tau_delta - tau_T| < 0.1 μs` (lifetimes too similar)
3. `R² < 0.85` (poor fit)
4. Fit fails to converge

**Fitting Models:**

##### Main Model: `f(t-t0)`
```python
# Biexponential
I(t) = A * [exp(-(t-t0)/tau_delta) - exp(-(t-t0)/tau_T)] + y0
# For t < t0: I(t) = y0

# Single exponential
I(t) = A * exp(-(t-t0)/tau_delta) + y0
# For t < t0: I(t) = y0
```

**Parameters:**
- `A`: Amplitude (used for QY calculation)
- `tau_delta`: Singlet O2 lifetime (primary parameter)
- `tau_T`: Triplet rise time (biexponential only)
- `t0`: Lag time (time shift, fitted)
- `y0`: Baseline offset

**Fit Quality Metrics:**
```python
# R-squared
SS_res = sum((y_true - y_pred)**2)
SS_tot = sum((y_true - y_mean)**2)
R² = 1 - (SS_res / SS_tot)

# Chi-square
χ² = sum((residuals / sqrt(intensity))**2)
χ²_reduced = χ² / (n_points - n_parameters)
# Good fit: χ²_reduced ≈ 1.0
```

**Parameter Bounds:**
```python
# Biexponential
bounds = (
    [0,    0.5,  0.01,  -2.0,  -np.inf],  # Lower: [A, tau_delta, tau_T, t0, y0]
    [np.inf, 10.0, 5.0,   2.0,  np.inf]   # Upper
)

# Single exponential
bounds = (
    [0,    0.5,  -2.0,  -np.inf],  # Lower: [A, tau_delta, t0, y0]
    [np.inf, 10.0, 2.0,  np.inf]   # Upper
)
```

**Key Features:**
- **Automatic model selection:** Tries complex model first, falls back if needed
- **Parameter exchange detection:** Corrects if fitter swaps tau_delta ↔ tau_T
- **Robust error handling:** Catches divergence, invalid parameters
- **Comprehensive output:** Returns all data + fit + metrics in single dataclass

**Used By:** gui.analysis_worker, surplus.surplus_analyzer

---

### 8. statistical_analyzer.py
**Location:** `core/statistical_analyzer.py` (218 lines)
**Purpose:** Calculate statistics across replicates
**Status:** ✅ ACTIVE - Statistics engine

**Dependencies:**
```python
import numpy as np
from typing import List, Dict, Any
from core.kinetics_dataclasses import KineticsResult
```

**Exports:**

#### Class: `StatisticalAnalyzer`

**Main Method:**
```python
def analyze_replicate_statistics(
    results: List[KineticsResult]
) -> Dict[str, Any]:
    """
    Calculate mean, SD, CV% for all parameters across replicates.

    Returns dict with keys:
    - 'parameters': {param_name: {'mean': float, 'sd': float, 'cv_percent': float}}
    - 'fit_quality': {...}
    - 'n_replicates': int
    - 'outliers': List[int]  # Indices of outlier replicates
    """
```

**Calculated Statistics:**

For each parameter (A, tau_delta, tau_T, t0, y0, R², χ²_r, SNR):
```python
mean = np.mean(values)
sd = np.std(values, ddof=1)  # Sample SD (n-1)
cv_percent = (sd / mean) * 100  # Coefficient of variation
```

**Outlier Detection:**
```python
def detect_outliers(values: np.ndarray, threshold: float = 2.5) -> List[int]:
    """
    Outliers = |value - median| > threshold * MAD
    MAD = Median Absolute Deviation (robust to outliers)

    Returns: List of outlier indices
    """
```

**Output Example:**
```python
{
    'parameters': {
        'A': {'mean': 1250.5, 'sd': 45.2, 'cv_percent': 3.6},
        'tau_delta': {'mean': 3.48, 'sd': 0.12, 'cv_percent': 3.4},
        'tau_T': {'mean': 1.87, 'sd': 0.08, 'cv_percent': 4.3},
        't0': {'mean': 0.15, 'sd': 0.05, 'cv_percent': 33.3},
        'y0': {'mean': 12.3, 'sd': 5.1, 'cv_percent': 41.5}
    },
    'fit_quality': {
        'r_squared': {'mean': 0.997, 'sd': 0.002, 'cv_percent': 0.2},
        'reduced_chi_square': {'mean': 1.05, 'sd': 0.12, 'cv_percent': 11.4}
    },
    'snr': {
        'snr_linear': {'mean': 52.6, 'sd': 8.3, 'cv_percent': 15.8}
    },
    'n_replicates': 3,
    'outliers': []
}
```

**Used By:** gui.analysis_worker

---

### 9. quantum_yield_calculator.py
**Location:** `core/quantum_yield_calculator.py` (238 lines)
**Purpose:** Quantum yield calculations
**Status:** ✅ ACTIVE - QY calculator

**Dependencies:**
```python
import numpy as np
from typing import List, Dict, Tuple, Any
from core.kinetics_dataclasses import KineticsResult
from utils.logger_config import get_logger
```

**Exports:**

#### Function: `calculate_quantum_yields_simple`

**Signature:**
```python
def calculate_quantum_yields_simple(
    kinetics_results: Dict[str, List[KineticsResult]],
    standards: List[str] = None,  # Auto-detect if None
    samples: List[str] = None     # Auto-detect if None
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Calculate quantum yields using A-based method.

    ONE METHOD ONLY: A-based (amplitude method)
    Formula: QY_sample = QY_standard × (A_sample / A_standard) × (Abs_standard / Abs_sample)

    Returns:
    - qy_pairs: List of dicts describing each standard-sample pair
    - qy_results: Dict[sample_name] = {'qy_mean': float, 'qy_sd': float, ...}
    """
```

**Algorithm:**

##### 1. Auto-detect standards and samples
```python
# Standard = has QY value in filename (QY parameter present)
# Sample = no QY value in filename (QY parameter absent)

standards = [compound for compound, results in kinetics_results.items()
             if results[0].quantum_yield is not None]

samples = [compound for compound, results in kinetics_results.items()
           if results[0].quantum_yield is None]
```

##### 2. Pair by wavelength
```python
# For each sample, find standards with matching excitation wavelength
for sample in samples:
    sample_wavelength = kinetics_results[sample][0].wavelength

    matching_standards = [std for std in standards
                          if kinetics_results[std][0].wavelength == sample_wavelength]
```

##### 3. Calculate QY for each replicate
```python
# For each sample replicate:
QY_sample = QY_standard × (A_sample / A_standard) × (Abs_standard / Abs_sample)
```

**Where:**
- `A_sample`: Amplitude from `FitParameters.A` (main model)
- `A_standard`: Amplitude from standard
- `Abs_sample`: Absorbance at excitation wavelength (from Abs file)
- `Abs_standard`: Absorbance at excitation wavelength
- `QY_standard`: Literature value (from filename, e.g., `QY0.98`)

##### 4. Average across replicates
```python
qy_mean = np.mean(qy_values)
qy_sd = np.std(qy_values, ddof=1)
```

**Output Example:**
```python
# qy_pairs
[
    {
        'standard': 'Phenalenone_400nm',
        'sample': 'TMPyP_400nm',
        'wavelength': 400.0,
        'qy_standard': 0.98,
        'n_sample_replicates': 3,
        'n_standard_replicates': 3
    },
    ...
]

# qy_results
{
    'TMPyP_400nm': {
        'qy_mean': 0.75,
        'qy_sd': 0.03,
        'qy_values': [0.73, 0.76, 0.76],  # Per replicate
        'standard_used': 'Phenalenone_400nm',
        'n_replicates': 3
    },
    ...
}
```

**Key Notes:**
- ❌ NO other QY methods (AUC, S0-based) — removed from SOLIS
- ✅ ONLY A-based method (amplitude method)
- Requires absorbance data (Abs files) linked to decay files
- Auto-pairing by wavelength

**Used By:** gui.analysis_worker

---

## GUI Components

### 1. integrated_browser_widget.py
**Location:** `gui/integrated_browser_widget.py` (1374 lines)
**Purpose:** File browser + data selector + SNR display
**Status:** ✅ ACTIVE - Critical GUI component

**Dependencies:**
```python
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTreeWidget,
    QTreeWidgetItem, QLabel, QCheckBox, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from data.file_parser import FileParser, ParsedFile
from gui.dataset_classification_dialog import classify_dataset
```

**Exports:**

#### Class: `IntegratedBrowserWidget(QWidget)`

The main data browser for file loading and selection.

**Signals:**
```python
folder_loaded = pyqtSignal(dict)  # Emits parsed compounds
selection_changed = pyqtSignal(dict)  # Emits selected compounds + replicates
snr_calculated = pyqtSignal(dict)  # Emits SNR results
```

**Key Components:**

##### Tree View
```
📁 Data Folder: /path/to/data/
  📦 TMPyP_400nm (Standard, QY=0.98)
    ├── ✓ Rep 1 | SNR: 52.6:1 | Quality: Excellent | Dataset: lag_spike
    ├── ✓ Rep 2 | SNR: 48.3:1 | Quality: Excellent | Dataset: lag_spike
    └── ✓ Rep 3 | SNR: 55.1:1 | Quality: Excellent | Dataset: lag_spike
  📦 Porphyrin_450nm (Sample)
    ├── ✓ Rep 1 | SNR: 12.4:1 | Quality: Good | Dataset: spike_only
    └── ✗ Rep 2 | SNR: 2.8:1 | Quality: Poor | Dataset: spike_only
```

##### Checkboxes
- Each replicate has checkbox (✓ = include in analysis, ✗ = exclude)
- Compounds with all replicates unchecked are excluded

##### SNR Display
- Real-time SNR calculation when folder loaded
- Color-coded quality:
  - 🟢 Excellent (green)
  - 🔵 Good (blue)
  - 🟡 Fair (yellow)
  - 🔴 Poor (red)

##### Dataset Classification
- Auto-detects dataset type or prompts user
- Types: `lag_spike`, `spike_only`, `clean_signal`, `preprocessed`

**Main Methods:**

```python
def load_folder(folder_path: str):
    """
    1. Parse all CSV files using FileParser
    2. Link absorbance data
    3. Validate file structure
    4. Populate tree widget
    5. Calculate SNR for all replicates (background thread)
    6. Display in tree with checkboxes
    """

def get_selected_compounds() -> Dict[str, List[ParsedFile]]:
    """
    Returns dict of selected compounds with only checked replicates.
    Used by analysis_worker to know what to analyze.
    """

def update_snr_display(snr_results: Dict[str, Dict[int, SNRResult]]):
    """
    Update tree widget with SNR values and quality labels.
    Called when SNR calculation completes.
    """

def apply_snr_filter(threshold: float):
    """
    Uncheck replicates with SNR below threshold.
    Called from preferences or toolbar.
    """
```

**Workflow:**
```
User clicks "Load Folder"
  ↓
FileParser.parse_directory()
  ↓
Tree populated with compounds/replicates
  ↓
Background: Calculate SNR for all
  ↓
Display SNR + quality in tree
  ↓
User checks/unchecks replicates
  ↓
Emit selection_changed signal
  ↓
MainWindow receives selected data
```

**Key Features:**
- **Lazy SNR calculation:** Only calculates when folder loaded (not on every file)
- **Dataset classification:** Interactive dialog if auto-detection uncertain
- **SNR filtering:** Quick filter to exclude low-quality data
- **Absorbance linking:** Auto-links Abs files to Decay files by compound name

**Used By:** solis_gui.SOLISMainWindow (central widget)

---

### 2. analysis_worker.py
**Location:** `gui/analysis_worker.py` (374 lines)
**Purpose:** Background analysis thread (QThread)
**Status:** ✅ ACTIVE - Critical worker thread

**Dependencies:**
```python
from PyQt6.QtCore import QThread, pyqtSignal
from core.snr_analyzer import SNRAnalyzer
from core.kinetics_analyzer import KineticsAnalyzer
from core.statistical_analyzer import StatisticalAnalyzer
from core.quantum_yield_calculator import calculate_quantum_yields_simple
from utils.logger_config import get_logger
```

**Exports:**

#### Class: `AnalysisWorker(QThread)`

Runs complete analysis pipeline in background to keep GUI responsive.

**Initialization:**
```python
worker = AnalysisWorker(
    compounds_data=selected_compounds,  # From integrated_browser
    snr_threshold=5.0,                  # Minimum SNR
    mask_corrections={}                 # Optional manual mask overrides
)
```

**Signals:**
```python
progress_update = pyqtSignal(str, int)  # (message, percentage)
snr_calculated = pyqtSignal(dict)       # SNR results
replicate_analyzed = pyqtSignal(str, int, object)  # (compound, rep_idx, KineticsResult)
analysis_complete = pyqtSignal(dict, dict, dict)   # (kinetics, statistics, qy_results)
error_occurred = pyqtSignal(str)        # Error message
```

**Analysis Pipeline:**

```python
def run(self):
    """
    Complete analysis pipeline (runs in separate thread):

    1. Calculate SNR for all replicates
       - Progress: 0-20%
       - Emit: snr_calculated

    2. Filter by SNR threshold
       - Exclude replicates below threshold

    3. Fit kinetics for each replicate
       - Progress: 20-80%
       - Emit: replicate_analyzed (for each replicate)

    4. Calculate statistics per compound
       - Progress: 80-90%
       - Mean, SD, CV% for all parameters

    5. Calculate quantum yields
       - Progress: 90-100%
       - Auto-pair standards with samples

    6. Emit analysis_complete
       - Returns: kinetics_results, statistics, qy_results
    """
```

**Step-by-Step:**

##### Step 1: SNR Calculation
```python
snr_results = {}
for compound, parsed_files in compounds_data.items():
    snr_results[compound] = {}
    for rep_idx, parsed_file in enumerate(parsed_files):
        time, intensity = parsed_file.get_kinetics_data()

        snr_analyzer = SNRAnalyzer(dataset_type=parsed_file.dataset_type)
        snr_result = snr_analyzer.analyze_snr(time[rep_idx], intensity[rep_idx])

        snr_results[compound][rep_idx] = snr_result

emit snr_calculated(snr_results)
```

##### Step 2: Filtering
```python
# Exclude replicates with SNR < threshold
filtered_compounds = {}
for compound, parsed_files in compounds_data.items():
    filtered_compounds[compound] = []
    for rep_idx, parsed_file in enumerate(parsed_files):
        if snr_results[compound][rep_idx].snr_linear >= snr_threshold:
            filtered_compounds[compound].append((rep_idx, parsed_file))
```

##### Step 3: Kinetics Fitting
```python
kinetics_results = {}
analyzer = KineticsAnalyzer()

for compound, rep_list in filtered_compounds.items():
    kinetics_results[compound] = []

    for rep_idx, parsed_file in rep_list:
        time, intensity = parsed_file.get_kinetics_data()

        # Check for manual mask override
        mask_key = f"{compound}_Rep{rep_idx+1}"
        custom_mask = mask_corrections.get(mask_key, None)

        result = analyzer.fit_kinetics(
            time[rep_idx],
            intensity[rep_idx],
            tau_delta_fixed=parsed_file.tau_delta_fixed,
            dataset_type=parsed_file.dataset_type,
            custom_mask_end_us=custom_mask
        )

        kinetics_results[compound].append(result)
        emit replicate_analyzed(compound, rep_idx, result)
```

##### Step 4: Statistics
```python
statistics = {}
stat_analyzer = StatisticalAnalyzer()

for compound, results_list in kinetics_results.items():
    if len(results_list) > 1:  # Need at least 2 replicates for stats
        statistics[compound] = stat_analyzer.analyze_replicate_statistics(results_list)
```

##### Step 5: Quantum Yields
```python
qy_pairs, qy_results = calculate_quantum_yields_simple(kinetics_results)
```

##### Step 6: Complete
```python
emit analysis_complete(kinetics_results, statistics, qy_results)
```

**Key Features:**
- **Non-blocking:** Runs in separate thread (GUI remains responsive)
- **Progress reporting:** Real-time updates for user feedback
- **Error handling:** Catches exceptions, emits error_occurred signal
- **Incremental results:** Emits replicate_analyzed for live plot updates

**Threading Pattern:**
```python
# In main GUI
worker = AnalysisWorker(data, threshold)
worker.progress_update.connect(self.update_progress_bar)
worker.replicate_analyzed.connect(self.display_replicate_plot)
worker.analysis_complete.connect(self.display_final_results)
worker.error_occurred.connect(self.show_error_dialog)
worker.start()  # Runs in background
```

**Used By:** solis_gui.SOLISMainWindow

---

### 3. splash_screen.py
**Location:** `gui/splash_screen.py` (240 lines)
**Purpose:** Animated loading screen with progress bar
**Status:** ✅ ACTIVE - Application startup

**Dependencies:**
```python
from PyQt6.QtWidgets import QSplashScreen, QLabel, QProgressBar, QWidget, QVBoxLayout
from PyQt6.QtGui import QPixmap, QMovie
from PyQt6.QtCore import Qt, QTimer
```

**Exports:**

#### Class: `SOLISSplashScreen(QSplashScreen)`

**Features:**
- SOLIS logo (JPG, 800x271 px)
- Animated spinner (GIF)
- Progress bar (0-100%)
- Status messages ("Loading modules...", "Initializing...")

**Methods:**
```python
def __init__(self):
    """Load logo, setup spinner and progress bar."""

def setProgress(self, value: int):
    """Update progress bar (0-100)."""

def showMessage(self, message: str):
    """Display status message."""
```

**Usage:**
```python
# In show_splash_then_load.py
splash = SOLISSplashScreen()
splash.show()
splash.setProgress(0)
splash.showMessage("Loading SOLIS...")

# ... heavy imports ...

splash.setProgress(50)
splash.showMessage("Loading modules...")

# ... create main window ...

splash.setProgress(100)
splash.showMessage("Ready!")
splash.finish(main_window)  # Close splash, show main window
```

**Used By:** show_splash_then_load.py (main entry point)

---

### 4-9. Other GUI Components (Summary)

#### 4. **plot_viewer_widget.py** (1176 lines)
- Matplotlib canvas for displaying plots
- Zoom, pan, save to file
- Used for replicate plots, statistics plots, QY plots

#### 5. **preferences_dialog.py** (244 lines)
- Settings dialog (Edit > Preferences)
- SNR thresholds (homogeneous, heterogeneous)
- Surplus method parameters
- Saves to solis_gui preferences dict

#### 6. **dataset_classification_dialog.py** (207 lines)
- Dialog for classifying dataset types
- Called when auto-detection is uncertain
- Options: lag_spike, spike_only, clean_signal, preprocessed

#### 7. **heterogeneous_dialog.py** (562 lines)
- Dialog for heterogeneous analysis
- Parameter inputs (tau_T range, tau_delta_W range, tau_delta_L range)
- Grid search controls (resolution, max time)
- Surplus method toggle
- Displays results with plot

#### 8. **variable_study_widget.py** (705 lines)
- Linearity check analysis widget
- Plots parameter vs absorbance or excitation energy
- Linear regression with R², slope, intercept
- Used for validating experimental setup

---

## Data Layer

### file_parser.py
**Location:** `data/file_parser.py` (596 lines)
**Purpose:** Parse CSV files with comprehensive validation
**Status:** ✅ ACTIVE - Critical data parser

**Dependencies:**
```python
import re
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
```

**Exports:**

#### `@dataclass ParsedFile`
Container for parsed file data.
```python
@dataclass
class ParsedFile:
    compound: str                          # e.g., "TMPyP", "Rose.Bengal"
    file_type: str                         # 'decay' or 'absorption'
    file_path: str                         # Full path
    wavelength: float                      # Excitation wavelength (nm)
    tau_delta_fixed: Optional[float]       # Fixed tau_delta (if specified)
    quantum_yield: Optional[float]         # QY value (Standards only)
    quantum_yield_sd: Optional[float]      # QY uncertainty
    excitation_energy_mj: Optional[float]  # Excitation energy (mJ)
    excitation_energy_unit: Optional[str]  # 'mJ' or 'uJ'
    classification: str                    # 'Standard' or 'Sample'
    absorbance_at_wavelength: float | List[float]  # From Abs file
    dataset_type: str                      # 'lag_spike', 'spike_only', etc.
    data: pd.DataFrame                     # Time/Intensity columns

    def get_kinetics_data(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Returns: (time, [intensity_rep1, intensity_rep2, ...])
        Handles 1 to N replicates.
        """

    def get_absorbance_for_replicate(self, index: int) -> float:
        """Get absorbance for specific replicate (if list) or single value."""
```

#### Class: `FileParseError(Exception)`
Custom exception for parsing errors.

#### Class: `FileParser`

**Main Methods:**

##### Parse Directory
```python
def parse_directory(self, directory: str) -> Dict[str, List[ParsedFile]]:
    """
    Parse all CSV files in directory.

    Returns: Dict[compound_name, List[ParsedFile]]

    Algorithm:
    1. Find all CSV files
    2. Classify as Decay or Absorption
    3. Parse filename patterns
    4. Read CSV data
    5. Group by compound
    """
```

##### Parse Single File
```python
@lru_cache(maxsize=50)  # Cache for performance
def parse_file(self, file_path: str) -> ParsedFile:
    """
    Parse single CSV file.

    Steps:
    1. Match filename pattern (Decay_... or Abs_...)
    2. Extract metadata (compound, wavelength, QY, etc.)
    3. Detect CSV delimiter (auto)
    4. Read data into DataFrame
    5. Validate structure
    6. Return ParsedFile
    """
```

##### Link Absorption Data
```python
def link_absorption_data(
    self,
    parsed_files: List[ParsedFile],
    directory: str
) -> None:
    """
    Link Abs files to Decay files by compound name and wavelength.
    Updates ParsedFile.absorbance_at_wavelength in-place.

    Matching logic:
    - Same compound name (case-insensitive, dots allowed)
    - Decay wavelength exists in Abs spectrum
    - Interpolates if needed
    """
```

##### Validate File Structure
```python
def validate_file_structure(self, directory: str) -> Dict[str, Any]:
    """
    Comprehensive validation of data folder.

    Checks:
    - All Decay files have matching Abs files
    - All compounds have required wavelengths
    - Standards have QY values
    - File naming conventions correct
    - Data columns present (Time, Intensity, Replicate)

    Returns: Dict with 'valid': bool, 'errors': List[str], 'warnings': List[str]
    """
```

**Filename Patterns:**

##### Decay Files
```regex
Decay_[Compound]_EX[λ]nm_tauD[value]_QY[value]_QYsd[sd]_EI[value][unit].csv

Examples:
- Decay_TMPyP_EX400nm_tauD3.5_QY0.98_QYsd0.08.csv (Standard with QY)
- Decay_Porphyrin_EX450nm_tauD3.5.csv (Sample, no QY)
- Decay_Rose.Bengal_EX532nm_tauD3.5_EI2.5mJ.csv (With excitation energy)
```

Components:
- `Compound`: Any name (dots allowed for "Rose.Bengal")
- `EX[λ]nm`: Excitation wavelength (required)
- `tauD[value]`: Fixed tau_delta (optional, default 3.5)
- `QY[value]`: Quantum yield (optional, marks as Standard)
- `QYsd[sd]`: QY uncertainty (optional)
- `EI[value][unit]`: Excitation energy (optional, mJ or uJ)

##### Absorption Files
```regex
Abs_[Compound].csv

Examples:
- Abs_TMPyP.csv
- Abs_Rose.Bengal.csv
```

**CSV Structure:**

##### Decay File (1 replicate)
```csv
Time,Intensity
0.00,10.5
0.02,12.3
0.04,15.8
...
```

##### Decay File (3 replicates)
```csv
Time,Intensity_Rep1,Intensity_Rep2,Intensity_Rep3
0.00,10.5,11.2,10.8
0.02,12.3,12.9,12.1
0.04,15.8,16.2,15.5
...
```

##### Absorption File
```csv
Wavelength,Absorbance
300,0.05
350,0.15
400,0.45
450,0.32
...
```

**Key Features:**
- **Flexible delimiters:** Auto-detects `,`, `\t`, `;`
- **LRU cache:** Caches 50 most recent files for performance
- **Robust parsing:** Handles dots in compound names, optional parameters
- **Comprehensive validation:** Catches common errors (missing files, mismatched names)

**Used By:** gui.integrated_browser_widget

---

## Plotting System

### 1. solis_plotter.py
**Location:** `plotting/solis_plotter.py` (962 lines)
**Purpose:** Publication-quality plotting for homogeneous analysis
**Status:** ✅ ACTIVE - Main plotter

**Dependencies:**
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from core.kinetics_dataclasses import KineticsResult
from typing import Optional, List
```

**Exports:**

#### Class: `SOLISPlotter`

**Initialization:**
```python
plotter = SOLISPlotter(
    output_dir='plots/',
    dpi=300,                    # High-res for publication
    figure_size=(10, 8),
    font_family='Arial',
    font_size=12
)
```

**Main Methods:**

##### Single Replicate Plot
```python
def plot_single_replicate(
    self,
    result: KineticsResult,
    log_x: bool = True,
    show_legend: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create 3-panel plot:
    1. Decay curve (75% height)
       - Black dots: Experimental data
       - Red line: Fitted curve
       - Vertical line at t0
       - Shaded spike region
    2. Weighted residuals (12.5% height)
       - Symmetrical y-axis (±max_residual)
       - Horizontal line at y=0
    3. Parameters table (12.5% height)
       - A, tau_delta, tau_T, t0, y0
       - R², χ²_r, SNR

    Returns: Figure object
    """
```

**Plot Layout:**
```
+---------------------------+
|                           |
|   Decay Curve (log-x)     |  75% height
|   • Data  ─ Fit           |
|   | spike |  signal       |
|                           |
+---------------------------+
| Weighted Residuals        |  12.5%
| ─────────0────────────    |
+---------------------------+
| Fit Parameters Table      |  12.5%
| A, τΔ, τT, R², SNR        |
+---------------------------+
```

##### Batch Plot (Multiple Replicates)
```python
def plot_batch(
    self,
    results: List[KineticsResult],
    layout: str = 'grid',  # 'grid' or 'overlay'
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple replicates.

    Layouts:
    - 'grid': Separate subplots (2x2, 3x3, etc.)
    - 'overlay': All on same axes (different colors)
    """
```

##### Statistics Plot
```python
def plot_statistics(
    self,
    statistics: Dict[str, Any],
    compound_name: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar plots with error bars for mean ± SD:
    - Amplitude
    - tau_delta
    - tau_T
    - R²
    - SNR
    """
```

##### Quantum Yield Plot
```python
def plot_quantum_yields(
    self,
    qy_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar chart of QY values with error bars.
    Color-coded by sample.
    """
```

**Export Methods:**
```python
def export_pdf(self, fig: plt.Figure, filename: str):
    """Save as vector PDF (publication quality)."""

def export_png(self, fig: plt.Figure, filename: str, dpi: int = 300):
    """Save as high-res PNG."""

def export_svg(self, fig: plt.Figure, filename: str):
    """Save as SVG (editable in Illustrator)."""
```

**Styling:**
```python
# Default style (publication-ready)
- Font: Arial 12pt
- Line width: 1.5 pt (data), 2.0 pt (fit)
- Marker size: 3 pt
- Grid: Light gray, alpha=0.3
- Colors: Black (data), Red (fit), Blue (residuals)
- Spine visibility: Top/right hidden
```

**Used By:** gui.plot_viewer_widget, solis_gui

---

### 2. variable_study_plotter.py
**Location:** `plotting/variable_study_plotter.py` (706 lines)
**Purpose:** Linearity check plots (concentration, excitation energy)
**Status:** ✅ ACTIVE - Linearity plotter

**Dependencies:**
```python
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np
```

**Exports:**

#### Class: `VariableStudyPlotter`

**Plot Types:**

##### 1. Parameter vs Absorbance
```python
def plot_parameter_vs_absorbance(
    self,
    absorbances: List[float],
    parameter_values: List[float],
    parameter_name: str,          # 'A', 'tau_delta', etc.
    compound_name: str,
    wavelength: float
) -> go.Figure:
    """
    Linear plot: parameter vs absorbance.
    Includes:
    - Scatter points with error bars
    - Linear regression line
    - R², slope, intercept
    - 95% confidence interval (shaded)
    """
```

##### 2. Parameter vs Excitation Energy
```python
def plot_parameter_vs_excitation_energy(
    self,
    excitation_energies: List[float],
    parameter_values: List[float],
    parameter_name: str,
    compound_name: str,
    wavelength: float
) -> go.Figure:
    """Similar to absorbance plot, but x-axis = excitation energy (mJ)."""
```

##### 3. Multi-Parameter Dashboard
```python
def plot_linearity_dashboard(
    self,
    x_values: List[float],
    x_label: str,
    parameters: Dict[str, List[float]],  # {param_name: values}
    compound_name: str
) -> go.Figure:
    """
    4x2 subplot grid for multiple parameters:
    - A vs x
    - tau_delta vs x
    - tau_T vs x
    - R² vs x
    - SNR vs x
    - chi_square vs x
    """
```

**Linear Regression:**
```python
# Fit: y = slope * x + intercept
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(x.reshape(-1, 1), y)
r_value, p_value = pearsonr(x, y)

# Display on plot
annotations = [
    f"R² = {r_squared:.4f}",
    f"Slope = {slope:.4f}",
    f"Intercept = {intercept:.4f}",
    f"p-value = {p_value:.4e}"
]
```

**Acceptance Criteria:**
```python
# Good linearity
R² > 0.95          # Strong correlation
p_value < 0.05     # Statistically significant

# Check parameters
- A should increase linearly with absorbance/energy
- tau_delta should be constant (slope ~ 0)
- tau_T should be constant (slope ~ 0)
- R² should be consistently high (> 0.95)
```

**Used By:** gui.variable_study_widget

---

## Heterogeneous Analysis

### 1. heterogeneous_dataclasses.py
**Location:** `heterogeneous/heterogeneous_dataclasses.py` (217 lines)
**Purpose:** Data structures for vesicle/membrane analysis
**Status:** ✅ ACTIVE - Core data structures

**Exports:**

#### `@dataclass VesicleGeometry`
Defines vesicle structure for diffusion simulation.
```python
@dataclass
class VesicleGeometry:
    diameter_nm: float = 78.0         # Total vesicle diameter
    membrane_thickness_nm: float = 4.0  # Lipid bilayer thickness
    grid_spacing_nm: float = 1.0      # Spatial resolution

    # Computed properties
    @property
    def radius_nm(self) -> float:
        return self.diameter_nm / 2.0

    @property
    def n_layers(self) -> int:
        """Number of concentric shells."""
        return int(self.radius_nm / self.grid_spacing_nm) + 1

    @property
    def membrane_start_layer(self) -> int:
        """First layer of membrane."""
        return int((self.radius_nm - self.membrane_thickness_nm) / self.grid_spacing_nm)

    @property
    def membrane_end_layer(self) -> int:
        """Last layer of membrane."""
        return self.n_layers - 1
```

#### `@dataclass DiffusionParameters`
Physical parameters for diffusion simulation.
```python
@dataclass
class DiffusionParameters:
    tau_T: float                 # Triplet decay time (μs)
    tau_delta_W: float           # ¹O₂ lifetime in water (μs)
    tau_delta_L: float           # ¹O₂ lifetime in lipid (μs)
    D_W: float = 2.5e-5          # Diffusion coeff in water (cm²/s)
    D_L: float = 1.0e-6          # Diffusion coeff in lipid (cm²/s)
    k_T: float = None            # Triplet decay rate (1/μs) [computed]
    k_delta_W: float = None      # ¹O₂ decay rate in water [computed]
    k_delta_L: float = None      # ¹O₂ decay rate in lipid [computed]

    def __post_init__(self):
        """Compute decay rates from lifetimes."""
        self.k_T = 1.0 / self.tau_T
        self.k_delta_W = 1.0 / self.tau_delta_W
        self.k_delta_L = 1.0 / self.tau_delta_L
```

#### `@dataclass SimulationResult`
Output from diffusion simulation.
```python
@dataclass
class SimulationResult:
    time_us: np.ndarray           # Time axis
    intensity: np.ndarray         # Simulated signal
    triplet_profile: np.ndarray   # [time, layer] - triplet concentration
    singlet_profile: np.ndarray   # [time, layer] - ¹O₂ concentration
    geometry: VesicleGeometry
    parameters: DiffusionParameters
```

#### `@dataclass HeterogeneousFitResult`
Best fit result from grid search.
```python
@dataclass
class HeterogeneousFitResult:
    # Best fit parameters
    tau_T: float
    tau_delta_W: float
    tau_delta_L: float

    # Fit quality
    r_squared: float
    chi_square: float
    reduced_chi_square: float

    # Data
    time_us: np.ndarray
    intensity_exp: np.ndarray
    intensity_fitted: np.ndarray
    residuals: np.ndarray
    weighted_residuals: np.ndarray

    # Grid search info
    n_evaluations: int            # Total parameter combinations tested
    grid_resolution: Tuple[int, int, int]  # (n_tau_T, n_tau_W, n_tau_L)
    computation_time_s: float
```

**Used By:** All heterogeneous modules

---

### 2. diffusion_simulator_numba.py
**Location:** `heterogeneous/diffusion_simulator_numba.py` (430 lines)
**Purpose:** Numba-accelerated 3D diffusion simulation
**Status:** ✅ ACTIVE - Core simulator

**Dependencies:**
```python
import numpy as np
import numba
from numba import jit, prange
from heterogeneous.heterogeneous_dataclasses import (
    VesicleGeometry, DiffusionParameters, SimulationResult
)
```

**Exports:**

#### Class: `DiffusionSimulatorNumba`

**Core Algorithm:**
Solves coupled reaction-diffusion equations on spherical grid.

**Equations:**
```
Triplet:      ∂[T]/∂t = -k_T * [T]
Singlet O₂:   ∂[¹O₂]/∂t = k_T * [T] - k_Δ * [¹O₂] + D * ∇²[¹O₂]
```

Where:
- `[T]`: Triplet concentration (localized in membrane)
- `[¹O₂]`: Singlet oxygen concentration (diffuses)
- `k_T`: Triplet decay rate
- `k_Δ`: ¹O₂ decay rate (region-dependent: k_Δ_W or k_Δ_L)
- `D`: Diffusion coefficient (region-dependent: D_W or D_L)
- `∇²`: Laplacian (spherical coordinates, radial only)

**Initialization:**
```python
simulator = DiffusionSimulatorNumba(
    geometry=VesicleGeometry(diameter_nm=78, membrane_thickness_nm=4),
    dt_us=0.000125,             # Time step (125 ns)
    output_dt_us=0.02,          # Output sampling (20 ns)
    max_time_us=30.0            # Simulation duration
)
```

**Main Method:**
```python
def simulate(self, parameters: DiffusionParameters) -> SimulationResult:
    """
    Run diffusion simulation.

    Steps:
    1. Initialize grids (triplet in membrane, singlet oxygen = 0)
    2. Time loop:
       a. Decay triplet: T(t+dt) = T(t) * exp(-k_T * dt)
       b. Generate ¹O₂: S(t+dt) += k_T * T(t) * dt
       c. Diffuse ¹O₂: S(t+dt) += D * ∇²S * dt
       d. Decay ¹O₂: S(t+dt) *= exp(-k_Δ * dt)
    3. Record luminescence = sum over all layers
    4. Return SimulationResult
    """
```

**Numba Acceleration:**
```python
@jit(nopython=True, parallel=True)
def _diffusion_step(
    singlet_grid,      # [n_layers]
    D_grid,            # [n_layers] - diffusion coeff per layer
    k_delta_grid,      # [n_layers] - decay rate per layer
    dt,
    dr
):
    """
    JIT-compiled diffusion step (Numba).
    ~100x faster than pure Python.

    Laplacian in spherical coordinates (radial only):
    ∇²f = (1/r²) * d/dr(r² * df/dr)
    """
    new_grid = singlet_grid.copy()

    for i in prange(1, len(singlet_grid) - 1):  # Parallel loop
        r = i * dr

        # Compute radial derivative
        df_dr = (singlet_grid[i+1] - singlet_grid[i-1]) / (2 * dr)

        # Compute second derivative
        d2f_dr2 = (singlet_grid[i+1] - 2*singlet_grid[i] + singlet_grid[i-1]) / (dr**2)

        # Laplacian
        laplacian = d2f_dr2 + (2.0 / r) * df_dr

        # Diffusion + decay
        new_grid[i] += D_grid[i] * laplacian * dt
        new_grid[i] *= np.exp(-k_delta_grid[i] * dt)

    return new_grid
```

**Boundary Conditions:**
- Center (r=0): Neumann (zero flux)
- Surface (r=R): Absorbing (¹O₂ escapes to bulk)

**Performance:**
- Typical simulation: ~1-5 seconds (with Numba)
- Grid search (1000 simulations): ~20-60 minutes

**Used By:** heterogeneous_fitter, grid_search

---

### 3. grid_search.py
**Location:** `heterogeneous/grid_search.py` (418 lines)
**Purpose:** Parameter optimization via exhaustive grid search
**Status:** ✅ ACTIVE - Optimization engine

**Dependencies:**
```python
import numpy as np
from typing import Tuple, Dict, List
from heterogeneous.diffusion_simulator_numba import DiffusionSimulatorNumba
from heterogeneous.heterogeneous_dataclasses import DiffusionParameters
from dataclasses import dataclass
```

**Exports:**

#### `@dataclass GridSearchParams`
Defines parameter search space.
```python
@dataclass
class GridSearchParams:
    tau_T_range: Tuple[float, float]          # (min, max) μs
    tau_delta_W_range: Tuple[float, float]
    tau_delta_L_range: Tuple[float, float]

    tau_T_n_points: int = 10                  # Grid resolution
    tau_delta_W_n_points: int = 10
    tau_delta_L_n_points: int = 10

    # Total evaluations = tau_T_n * tau_delta_W_n * tau_delta_L_n
    # Example: 10 x 10 x 10 = 1,000 simulations
```

#### Class: `GridSearch`

**Main Method:**
```python
def run_grid_search(
    self,
    experimental_time: np.ndarray,
    experimental_intensity: np.ndarray,
    grid_params: GridSearchParams,
    simulator: DiffusionSimulatorNumba
) -> HeterogeneousFitResult:
    """
    Exhaustive grid search over parameter space.

    Algorithm:
    1. Generate parameter grid (3D)
    2. For each parameter combination:
       a. Run simulation
       b. Interpolate to experimental time axis
       c. Calculate R², χ²
    3. Find parameter set with best R²
    4. Return HeterogeneousFitResult

    Returns: Best fit result with all data
    """
```

**Grid Generation:**
```python
tau_T_values = np.linspace(
    grid_params.tau_T_range[0],
    grid_params.tau_T_range[1],
    grid_params.tau_T_n_points
)

tau_delta_W_values = np.linspace(...)
tau_delta_L_values = np.linspace(...)

# Create 3D grid
for tau_T in tau_T_values:
    for tau_W in tau_delta_W_values:
        for tau_L in tau_delta_L_values:
            # Evaluate this parameter set
```

**Fit Quality Metrics:**
```python
# R-squared
r_squared = 1 - (SS_res / SS_tot)

# Chi-square (Poisson weighting)
chi_square = sum((residuals / sqrt(intensity_exp))**2)
reduced_chi_square = chi_square / (n_points - n_parameters)

# Best fit = highest R²
```

**Performance Optimization:**
- **Simulation caching:** LRU cache for repeated parameter sets
- **Parallel evaluation:** Could use multiprocessing (not currently implemented)
- **Coarse-to-fine:** Start with coarse grid (5x5x5), refine around best fit

**Typical Runtime:**
```
10x10x10 grid = 1,000 simulations
  @ 2 sec/simulation → ~30 minutes

20x20x20 grid = 8,000 simulations
  @ 2 sec/simulation → ~4.5 hours
```

**Used By:** heterogeneous_fitter

---

### 4. heterogeneous_fitter.py
**Location:** `heterogeneous/heterogeneous_fitter.py` (364 lines)
**Purpose:** High-level fitting interface
**Status:** ✅ ACTIVE - Fitting engine

**Dependencies:**
```python
from heterogeneous.diffusion_simulator_numba import DiffusionSimulatorNumba
from heterogeneous.grid_search import GridSearch, GridSearchParams
from heterogeneous.heterogeneous_dataclasses import (
    VesicleGeometry, DiffusionParameters, HeterogeneousFitResult
)
```

**Exports:**

#### Class: `HeterogeneousFitter`

**Initialization:**
```python
fitter = HeterogeneousFitter(
    geometry=VesicleGeometry(diameter_nm=78, membrane_thickness_nm=4),
    simulation_dt_us=0.000125,
    output_dt_us=0.02,
    max_time_us=30.0
)
```

**Main Method:**
```python
def fit_experimental_data(
    self,
    time_us: np.ndarray,
    intensity: np.ndarray,
    grid_params: GridSearchParams,
    progress_callback=None         # For GUI progress updates
) -> HeterogeneousFitResult:
    """
    Fit experimental data to heterogeneous diffusion model.

    Workflow:
    1. Create simulator
    2. Setup grid search
    3. Run grid search (with progress updates)
    4. Return best fit result
    """
```

**Progress Callback:**
```python
# For GUI integration
def progress_callback(current, total, message):
    progress_bar.setValue(int(100 * current / total))
    status_label.setText(message)

# Call from fitter
fitter.fit_experimental_data(
    time, intensity, grid_params,
    progress_callback=progress_callback
)
```

**Used By:** gui.heterogeneous_dialog

---

### 5. heterogeneous_plotter_new.py
**Location:** `heterogeneous/heterogeneous_plotter_new.py` (257 lines)
**Purpose:** Plot heterogeneous fit results
**Status:** ✅ ACTIVE - Plotter

**Exports:**

#### Function: `plot_heterogeneous_result`

**Signature:**
```python
def plot_heterogeneous_result(
    result: HeterogeneousFitResult,
    title: str = "Heterogeneous Fit",
    log_x: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create 2-panel plot:
    1. Decay curve (top 80%)
       - Black: Experimental data
       - Red: Fitted curve
       - Log-x scale
    2. Weighted residuals (bottom 20%)
       - Symmetrical y-axis
       - Horizontal line at y=0

    Displays:
    - tau_T, tau_delta_W, tau_delta_L
    - R², χ²_r
    - Grid search info (n_evaluations, time)
    """
```

**Plot Layout:**
```
+---------------------------+
| Heterogeneous Fit         |
| • Experimental  ─ Fitted  |  80% height
|                           |
+---------------------------+
| Weighted Residuals        |  20%
| ─────────0────────────    |
+---------------------------+
```

**Used By:** gui.heterogeneous_dialog

---

### 6-7. BACKUP Files (Move to /old/)

#### heterogeneous/grid_search_BACKUP_twostep.py (715 lines)
- **Status:** 🗑️ BACKUP
- **Purpose:** Old two-step grid search (coarse → fine)
- **Action:** Move to `/old/`

#### heterogeneous/heterogeneous_fitter_BACKUP_twostep.py (~400 lines)
- **Status:** 🗑️ BACKUP
- **Purpose:** Old two-step fitter
- **Action:** Move to `/old/`

---

## Surplus Analysis

### surplus_analyzer.py
**Location:** `surplus/surplus_analyzer.py` (416 lines)
**Purpose:** 4-step surplus analysis for heterogeneous systems
**Status:** ✅ ACTIVE - Surplus analyzer

**Dependencies:**
```python
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from core.kinetics_analyzer import KineticsAnalyzer
```

**Exports:**

#### `@dataclass SurplusResult`
```python
@dataclass
class SurplusResult:
    # Step 1: Late-time homogeneous fit
    tau_delta_late: float
    A_late: float

    # Step 2: Surplus signal
    surplus_intensity: np.ndarray
    surplus_peak_time: float

    # Step 3: Surplus fit (early-time homogeneous)
    tau_delta_surplus: float
    A_surplus: float

    # Step 4: Full heterogeneous fit
    tau_T: float
    tau_delta_W: float
    tau_delta_L: float

    # Quality
    r_squared: float
    chi_square: float

    # Data
    time_us: np.ndarray
    intensity_raw: np.ndarray
    intensity_fitted: np.ndarray
    intensity_late_fit: np.ndarray
    residuals: np.ndarray
```

#### Function: `analyze_surplus`

**Signature:**
```python
def analyze_surplus(
    time_us: np.ndarray,
    intensity: np.ndarray,
    mask_time_us: float = 6.0,      # Mask before this time
    late_time_start_us: float = 20.0  # Start of late-time region
) -> SurplusResult:
    """
    4-step surplus method for extracting heterogeneous components.
    """
```

**Algorithm:**

##### Step 1: Fit Late-Time Region (Homogeneous)
```python
# Use data from late_time_start_us onward (e.g., t > 20 μs)
# Fit single exponential: I(t) = A_late * exp(-t/tau_delta_late) + y0
# This captures bulk ¹O₂ decay (homogeneous component)

late_time_mask = time_us >= late_time_start_us
analyzer = KineticsAnalyzer()
late_fit_result = analyzer.fit_kinetics(
    time_us[late_time_mask],
    intensity[late_time_mask],
    model='single_exponential'
)

tau_delta_late = late_fit_result.parameters.tau_delta
A_late = late_fit_result.parameters.A
```

##### Step 2: Calculate Surplus Signal
```python
# Extrapolate late-time fit to full time range
late_fit_extrapolated = homogeneous_model(time_us, A_late, tau_delta_late, t0, y0)

# Surplus = Raw - Late fit
surplus = intensity - late_fit_extrapolated

# Surplus should show early-time rise/decay from vesicles
```

##### Step 3: Fit Surplus (Early Homogeneous Component)
```python
# Fit surplus signal (mask artifact region)
surplus_mask = time_us >= mask_time_us
surplus_fit = analyzer.fit_kinetics(
    time_us[surplus_mask],
    surplus[surplus_mask],
    model='biexponential'  # May have rise time
)

tau_delta_surplus = surplus_fit.parameters.tau_delta
tau_T_surplus = surplus_fit.parameters.tau_T
A_surplus = surplus_fit.parameters.A
```

##### Step 4: Full Heterogeneous Fit
```python
# Use surplus parameters as initial guess for heterogeneous model
# Fit full data to biexponential or diffusion model
# Extract tau_T, tau_delta_W, tau_delta_L

final_fit = fit_heterogeneous_model(
    time_us,
    intensity,
    initial_guess=(tau_T_surplus, tau_delta_late, tau_delta_surplus)
)
```

**Key Concept:**
Surplus method decomposes signal into:
- **Late-time component:** Bulk ¹O₂ (homogeneous)
- **Early-time component:** Vesicle-localized ¹O₂ (heterogeneous)

By fitting these separately, initial guesses for full heterogeneous fit are improved.

**Advantages:**
- Faster than full grid search
- Provides physical insight (separate components)
- Good initial guess for full fit

**Disadvantages:**
- Assumes simple decomposition (may not work for complex systems)
- Sensitive to mask_time and late_time_start choices

**Used By:** gui.heterogeneous_dialog (surplus mode toggle)

---

## Utils & Infrastructure

### 1. session_manager.py
**Location:** `utils/session_manager.py` (534 lines)
**Purpose:** Session save/load with JSON serialization
**Status:** ✅ ACTIVE - Session management

**Dependencies:**
```python
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, is_dataclass
from typing import Dict, Any
from PyQt6.QtWidgets import QFileDialog, QMessageBox
```

**Exports:**

#### Class: `NumpyEncoder(json.JSONEncoder)`
Custom JSON encoder for NumPy/pandas/dataclasses.
```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'__ndarray__': obj.tolist(), 'dtype': str(obj.dtype)}
        elif isinstance(obj, pd.DataFrame):
            return {'__dataframe__': obj.to_dict('list')}
        elif is_dataclass(obj):
            return {'__dataclass__': obj.__class__.__name__, **asdict(obj)}
        # ... handle other types ...
```

#### Class: `SessionManager`

**Session File Format:** `.solis.json`

**Structure:**
```json
{
  "metadata": {
    "version": "1.0",
    "created": "2025-11-02T14:30:00",
    "solis_version": "1.0.0",
    "description": "Experiment 2025-11-02: TMPyP QY measurement"
  },
  "data": {
    "folder_path": "/path/to/data/",
    "loaded_compounds": {
      "TMPyP_400nm": [
        {"compound": "TMPyP", "file_type": "decay", ...},
        ...
      ]
    }
  },
  "analysis": {
    "homogeneous": {
      "kinetics_results": {...},
      "statistics": {...},
      "quantum_yields": {...}
    },
    "heterogeneous": {...},
    "surplus": {...}
  },
  "preferences": {
    "snr_thresholds": {"homogeneous": 5.0, "heterogeneous": 50.0},
    "surplus": {"mask_time_us": 6.0}
  },
  "ui_state": {
    "selected_compounds": ["TMPyP_400nm", "Porphyrin_450nm"],
    "selected_replicates": {...},
    "plot_settings": {...}
  }
}
```

**Main Methods:**

##### Save Session
```python
def save_session(
    self,
    session_data: Dict[str, Any],
    file_path: str
) -> bool:
    """
    Save complete session to JSON file.

    Steps:
    1. Add metadata (version, timestamp)
    2. Serialize data (NumPy → list, dataclasses → dict)
    3. Write to file with indentation (human-readable)
    4. Return success status
    """
```

##### Load Session
```python
def load_session(self, file_path: str) -> Dict[str, Any]:
    """
    Load session from JSON file.

    Steps:
    1. Read JSON file
    2. Validate version compatibility
    3. Deserialize data (list → NumPy, dict → dataclasses)
    4. Return session_data dict

    Raises: SessionLoadError if incompatible or corrupted
    """
```

##### GUI Dialogs
```python
def save_session_dialog(parent_widget, session_data) -> bool:
    """Show save dialog, call save_session if user confirms."""

def load_session_dialog(parent_widget) -> Optional[Dict]:
    """Show open dialog, call load_session if user selects file."""
```

**Serialization Examples:**

##### NumPy Array
```python
# Before
array = np.array([1.0, 2.5, 3.7])

# JSON
{"__ndarray__": [1.0, 2.5, 3.7], "dtype": "float64"}

# After
array = np.array(json_obj['__ndarray__'], dtype=json_obj['dtype'])
```

##### Dataclass
```python
# Before
result = FitParameters(A=1250.5, tau_delta=3.48, ...)

# JSON
{
    "__dataclass__": "FitParameters",
    "A": 1250.5,
    "tau_delta": 3.48,
    ...
}

# After
result = FitParameters(**{k: v for k, v in json_obj.items() if k != '__dataclass__'})
```

**Key Features:**
- **Full state preservation:** All data, results, preferences, UI state
- **Human-readable:** JSON with indentation (can inspect/edit)
- **Version checking:** Warns if loading newer session format
- **Error handling:** Validates on load, provides error messages

**Used By:** solis_gui (File > Save/Load Session)

---

### 2. csv_exporter.py
**Location:** `utils/csv_exporter.py` (estimated ~200 lines)
**Purpose:** Export analysis results to CSV files
**Status:** ✅ ACTIVE - Export utility

**Dependencies:**
```python
import pandas as pd
from pathlib import Path
from typing import Dict, List
from core.kinetics_dataclasses import KineticsResult
```

**Exports:**

#### Class: `CSVExporter`

**Main Methods:**

##### Export Kinetics Results
```python
def export_kinetics_results(
    self,
    results: Dict[str, List[KineticsResult]],
    output_file: str
):
    """
    Export all fit parameters to CSV.

    Format:
    Compound,Replicate,A,tau_delta,tau_T,t0,y0,R²,chi²_r,SNR
    TMPyP_400nm,1,1250.5,3.48,1.87,0.15,12.3,0.997,1.05,52.6
    TMPyP_400nm,2,1285.2,3.52,1.82,0.18,11.8,0.996,1.08,48.3
    ...
    """
```

##### Export Statistics
```python
def export_statistics(
    self,
    statistics: Dict[str, Dict],
    output_file: str
):
    """
    Export mean/SD/CV for each compound.

    Format:
    Compound,Parameter,Mean,SD,CV%
    TMPyP_400nm,A,1267.8,17.4,1.4
    TMPyP_400nm,tau_delta,3.50,0.02,0.6
    ...
    """
```

##### Export Quantum Yields
```python
def export_quantum_yields(
    self,
    qy_results: Dict[str, Dict],
    output_file: str
):
    """
    Export QY values with uncertainties.

    Format:
    Sample,QY_mean,QY_SD,Standard_used,n_replicates
    Porphyrin_450nm,0.75,0.03,Phenalenone_450nm,3
    ...
    """
```

##### Export Batch (All)
```python
def export_batch_analysis(
    self,
    kinetics_results: Dict,
    statistics: Dict,
    qy_results: Dict,
    output_dir: str
):
    """
    Export all results to separate CSV files in output_dir:
    - kinetics_results.csv
    - statistics.csv
    - quantum_yields.csv
    """
```

**Used By:** solis_gui (File > Export Results)

---

## Entry Points

### 1. show_splash_then_load.py
**Location:** `show_splash_then_load.py` (42 lines)
**Purpose:** Main application entry point with splash screen
**Status:** ✅ ACTIVE - Primary entry point

**Workflow:**
```python
1. Create QApplication
2. Show splash screen immediately
3. Import solis_gui (heavy, takes ~30 sec on first run)
4. Create SOLISMainWindow
5. Show main window, close splash
6. Start event loop
```

**Usage:**
```bash
python show_splash_then_load.py
```

**Why splash first?**
- PyQt6 + NumPy + SciPy + Matplotlib imports take 20-30 seconds
- Splash provides user feedback during startup
- Prevents "application not responding" perception

---

### 2. solis_gui.py
**Location:** `solis_gui.py` (887 lines)
**Purpose:** Main application window
**Status:** ✅ ACTIVE - Core GUI controller

**Key Components:**

##### Menu Bar
```
File
  ├── Load Folder (Ctrl+O)
  ├── Save Session (Ctrl+S)
  ├── Load Session (Ctrl+Shift+O)
  ├── Export Results (Ctrl+E)
  └── Exit (Ctrl+Q)

Analysis
  ├── Run Analysis
  ├── Heterogeneous Analysis
  ├── Linearity Check
  └── Clear Results

Plots
  ├── Show All Plots
  ├── Close All Plots
  └── Export All Plots

View
  ├── Show/Hide Browser
  ├── Show/Hide Preview
  └── Full Screen (F11)

Tools
  └── Preferences

Help
  ├── Documentation
  ├── About SOLIS
  └── Report Issue
```

##### Toolbar
```
[Load Folder] [Run Analysis] [Export] [Preferences] [Help]
```

##### Central Widget
- `IntegratedBrowserWidget` (data browser)

##### Dock Widgets
- Plot preview (optional)
- Parameter table (optional)

##### Status Bar
- Progress bar
- Status messages
- Ready indicator

**Key Methods:**
```python
def _on_load_folder():
    """Load data folder, populate browser."""

def _on_run_analysis():
    """Start analysis_worker thread."""

def _on_analysis_complete(kinetics, stats, qy):
    """Display results, enable export."""

def _save_session():
    """Save current session to .solis.json."""

def _load_session():
    """Load session, restore state."""

def _export_results():
    """Export to CSV files."""
```

**Lifecycle:**
```
Application Start
  ↓
Show Splash
  ↓
Load solis_gui
  ↓
Create SOLISMainWindow
  ↓
Show Main Window
  ↓
User loads folder → Browser populated
  ↓
User runs analysis → Worker thread started
  ↓
Analysis complete → Results displayed
  ↓
User exports → CSV files saved
  ↓
User saves session → State preserved
  ↓
Application Exit
```

---

## Test Files

All test files should be moved to `/test/` directory.

### 1. analyze_data_parameters.py (143 lines)
- **Purpose:** Debug script for data parameter analysis
- **Features:** Data diagnostics, time step checking, peak finding
- **Action:** Move to `/test/`

### 2. test_simulation_vs_data.py (131 lines)
- **Purpose:** Test heterogeneous simulation against experimental data
- **Action:** Move to `/test/`

### 3. test_compare_v2_current.py (109 lines)
- **Purpose:** Compare old vs new diffusion simulator versions
- **Action:** Move to `/test/`

### 4. test_grid_search_bug.py (103 lines)
- **Purpose:** Debug grid search functionality
- **Action:** Move to `/test/`

### 5. test_session_loading.py (~50 lines)
- **Purpose:** Test session save/load
- **Action:** Move to `/test/`

### 6. test_single_step_grid.py (~100 lines)
- **Purpose:** Test single-step grid search
- **Action:** Move to `/test/`

---

## Files to Archive

### Move to `/old/`

#### 1. heterogeneous/grid_search_BACKUP_twostep.py (715 lines)
- **Purpose:** Old two-step grid search implementation
- **Reason:** Replaced by single-step grid_search.py
- **Action:** `mv heterogeneous/grid_search_BACKUP_twostep.py old/`

#### 2. heterogeneous/heterogeneous_fitter_BACKUP_twostep.py (~400 lines)
- **Purpose:** Old two-step fitter implementation
- **Reason:** Replaced by heterogeneous_fitter.py
- **Action:** `mv heterogeneous/heterogeneous_fitter_BACKUP_twostep.py old/`

---

## Critical Usage Patterns

### 1. Data Structures Rule
❌ **NEVER** use dictionaries:
```python
result['A']                   # WRONG!
result.get('tau_delta')       # WRONG!
```

✅ **ALWAYS** use dataclass attributes:
```python
result.parameters.A           # CORRECT
result.fit_quality.r_squared  # CORRECT
result.snr_result.snr_linear  # CORRECT
```

### 2. SNR Display Convention
❌ **NEVER** show dB to users:
```python
f"SNR: {snr_result.snr_db:.1f} dB"  # Users don't understand dB!
```

✅ **ALWAYS** show linear ratio:
```python
f"SNR: {snr_result.snr_linear:.1f}:1"  # "52.6:1" - clear!
```

### 3. Logging Rule
❌ **NEVER** use `print()`:
```python
print("Analysis complete")  # No timestamps, not configurable
```

✅ **ALWAYS** use logger:
```python
from utils.logger_config import get_logger
logger = get_logger(__name__)
logger.info("Analysis complete")
```

### 4. Quantum Yield Method
❌ **NO** other methods:
```python
qy_auc()          # Removed!
qy_s0_based()     # Removed!
```

✅ **ONLY** A-based method:
```python
from core.quantum_yield_calculator import calculate_quantum_yields_simple
qy_pairs, qy_results = calculate_quantum_yields_simple(kinetics_results)
```

### 5. File Parsing Workflow
```python
from data.file_parser import FileParser

# 1. Parse directory
parser = FileParser()
compounds = parser.parse_directory("data/")

# 2. Link absorption data
for compound, files in compounds.items():
    parser.link_absorption_data(files, "data/")

# 3. Access data
for parsed_file in compounds['TMPyP_400nm']:
    if parsed_file.file_type == 'decay':
        time, intensity_replicates = parsed_file.get_kinetics_data()
```

### 6. Analysis Pipeline
```python
from core.kinetics_analyzer import KineticsAnalyzer

# 1. Create analyzer
analyzer = KineticsAnalyzer()

# 2. Fit kinetics
result = analyzer.fit_kinetics(
    time,
    intensity,
    tau_delta_fixed=3.5,        # Optional
    dataset_type='lag_spike',   # Optional
    custom_mask_end_us=0.5      # Optional
)

# 3. Access results
print(f"A = {result.parameters.A:.1f}")
print(f"τΔ = {result.parameters.tau_delta:.2f} μs")
print(f"R² = {result.fit_quality.r_squared:.4f}")
print(f"SNR = {result.snr_result.snr_linear:.1f}:1")
```

### 7. GUI Threading Pattern
```python
from PyQt6.QtCore import QThread, pyqtSignal

class Worker(QThread):
    result_ready = pyqtSignal(object)

    def run(self):
        # Long computation
        result = heavy_computation()
        self.result_ready.emit(result)

# In main window
worker = Worker()
worker.result_ready.connect(self.handle_result)
worker.start()  # Non-blocking
```

---

## Summary

**SOLIS** is a mature, well-architected application with:

✅ **Clean structure:**
- 37 active files (core functionality)
- 6 test files (should be moved)
- 2 backup files (should be archived)

✅ **Strong architecture:**
- Clear separation: core, gui, data, plotting, heterogeneous, surplus, utils
- Dataclass-based (type-safe, no dictionaries)
- Modular dependencies (Level 0-6 hierarchy)
- Background threading (responsive GUI)

✅ **Comprehensive features:**
- Homogeneous analysis (biexponential/single exponential)
- Heterogeneous analysis (vesicle diffusion)
- Surplus method
- Quantum yield calculations
- SNR analysis with spike detection
- Linearity studies
- Session save/load
- Publication-quality plots
- CSV export

✅ **Code quality:**
- Extensive documentation (docstrings, comments)
- Error handling
- Logging throughout
- Performance optimization (Numba, LRU cache)

**Next Steps:**
1. Move 6 test files to `/test/`
2. Move 2 backup files to `/old/`
3. Continue development with this clean structure

**This reference should be used alongside BACKEND_REFERENCE.md for complete technical documentation.**

---

**END OF SOLIS COMPLETE REFERENCE**
