# SOLIS Backend Reference
## Complete Module Documentation

**Last Updated:** 2025-11-02
**Purpose:** Reference for GUI development - all backend modules, their properties, and dependencies
**Companion:** See `CODEBASE_ANALYSIS_2025-11-02.md` for comprehensive analysis report

---

## Module Dependency Graph

```
Level 0 (No dependencies):
├── logger_config.py          (logging only)
└── kinetics_dataclasses.py   (numpy only)

Level 1 (Depends on Level 0):
├── core_fitting.py            → numpy, pandas, scipy
└── spike_detector_last.py     → numpy

Level 2 (Depends on Level 0-1):
├── masking_methods.py         → core_fitting, spike_detector_last
└── snr_analyzer.py            → kinetics_dataclasses, spike_detector_last

Level 3 (Depends on Level 0-2):
├── kinetics_analyzer.py       → masking_methods, snr_analyzer, kinetics_dataclasses, logger_config
└── file_parser.py             → pandas, numpy

Level 4 (Depends on Level 0-3):
├── statistical_analyzer.py    → kinetics_dataclasses
├── quantum_yield_calculator.py → kinetics_dataclasses
└── heterogeneous.py           → numpy

Level 5 (Depends on Level 0-4):
├── solis_plotter.py           → kinetics_dataclasses, plotly
└── csv_exporter.py            → kinetics_dataclasses, pandas
```

---

## Core Modules (12 Files)

### 1. **kinetics_dataclasses.py** (615 lines)
**Purpose:** Type-safe data structures for analysis results

**Dependencies:**
- numpy
- dataclasses (built-in)
- typing (built-in)

**Note:** Expanded from 399 → 615 lines (Oct-Nov 2025). Now includes `HeterogeneousFitResult` dataclass.

**Exports:**
- `SNRResult` - SNR analysis results
- `FitParameters` - Fitted parameters (A, τΔ, τT, t0, y0)
- `FitQuality` - Goodness of fit metrics (R², χ², reduced χ²)
- `LiteratureModelResult` - Literature model f(t) results
- `WorkflowInfo` - Analysis workflow metadata
- `KineticsResult` - Complete analysis result container
- `result_to_dict()` - Legacy compatibility function

**Key Properties:**
```python
# SNRResult
.snr_db: float              # SNR in decibels
.snr_linear: float          # SNR as ratio (DISPLAY THIS!)
.quality: str               # 'Excellent', 'Good', 'Fair', 'Poor'
.spike_region: dict         # Spike detection info
.has_spike() -> bool        # Check if spike detected
.is_good_quality(threshold) # Check quality

# FitParameters
.A: float                   # Amplitude (for QY)
.tau_delta: float           # Singlet oxygen lifetime
.tau_T: float | 'ND'        # Triplet rise time or 'ND'
.t0: float                  # Lag time
.y0: float                  # Baseline offset
.is_single_exponential()    # Check model type

# FitQuality
.r_squared: float           # R² (0-1)
.chi_square: float          # χ² statistic
.reduced_chi_square: float  # χ²ᵣ (should be ~1.0)
.model_used: str            # 'biexponential' or 'single_exponential'
.is_good_fit(r2, chi2)      # Check quality

# KineticsResult
.parameters: FitParameters
.fit_quality: FitQuality
.snr_result: SNRResult
.time_experiment_us: ndarray
.intensity_raw: ndarray
.main_curve: ndarray
.literature: LiteratureModelResult
.fitting_mask: ndarray
.spike_region: ndarray
.get_signal_region_data()   # Get masked data
.summary_string()           # Human-readable summary
```

---

### 2. **logger_config.py** (81 lines)
**Purpose:** Consistent logging across all modules

**Dependencies:**
- logging (built-in)
- sys (built-in)
- pathlib (built-in)

**Exports:**
- `setup_logger(name, level, log_file)` - Create logger
- `get_logger(name)` - Get/create default logger
- `set_global_log_level(level)` - Set level for all loggers

**Usage:**
```python
from logger_config import get_logger

logger = get_logger(__name__)
logger.info("Analysis complete")
logger.warning("Low SNR detected")
logger.error("Fit failed")
```

---

### 3. **core_fitting.py** (167 lines)
**Purpose:** Basic fitting functions and parameter handling

**Dependencies:**
- numpy
- pandas
- scipy.optimize.curve_fit

**Exports:**
- `r2_score(y_true, y_pred)` - Robust R² calculation
- `CoreFittingMethods` class

**CoreFittingMethods Properties:**
```python
.tau_delta_default: float = 3.5
.fitted_t0: float | None

# Methods:
.load_data(filename)
.literature_biexponential_free(t, A, τΔ, τT, t0, y0)
.literature_biexponential_pure(t, A, τΔ, τT, y0)
.literature_biexponential_pure_fixed_tau_delta(t, A, τT, y0, τΔ_fixed)
.detect_and_correct_parameter_exchange(A, τΔ, τT, y0, expected_τΔ)
.calculate_weighted_residuals(residuals, intensities)
.calculate_auc(A, τΔ, τT, y0)
.calculate_quantum_yield(A_sample, A_ref, abs_sample, abs_ref, qy_ref)
```

**Key Algorithms:**
- **R² calculation** with numerical stability checks
- **Parameter exchange detection** (τΔ ↔ τT swap correction)
- **Weighted residuals** for Poisson statistics

---

### 4. **masking_methods.py** (204 lines)
**Purpose:** Spike masking with time-shift approach

**Dependencies:**
- numpy
- scipy.optimize
- core_fitting
- spike_detector_last

**Exports:**
- `MaskingMethods` class (extends CoreFittingMethods)

**Key Methods:**
```python
.find_spikes(intensity, threshold_factor)
.find_spikes_from_replicate_info(intensity, replicate_results)
.fit_with_masking(time, intensity, extra_points, initial_guess)
.scan_masking_range(time, intensity, max_extra_points, initial_guess)
.trim_data_to_pulse_start(time, intensity)
```

**Algorithm:**
1. Detect spike region using `TransitionBasedSpikeDetector`
2. Time shift: spike_start → t=0
3. Create boolean mask (spike = False, signal = True)
4. Fit on masked data (t0 ≥ 0.0)

---

### 5. **snr_analyzer.py** (270 lines)
**Purpose:** Signal-to-noise ratio analysis with spike detection

**Dependencies:**
- numpy
- spike_detector_last
- kinetics_dataclasses.SNRResult
- logging

**Exports:**
- `SNRAnalyzer` class

**Properties:**
```python
.spike_detector: TransitionBasedSpikeDetector

# Methods:
.analyze_snr(x_data, y_data) -> SNRResult
.get_recommendations(snr_result) -> List[str]
```

**Algorithm:**
1. Detect spikes and baselines
2. Extract signal region (after spike)
3. Calculate SNR = (peak - baseline) / noise_std
4. Use **tail baseline** for better statistics
5. Assess quality: Excellent (>20dB), Good (>10dB), Fair (>3dB), Poor (≤3dB)

**Display Convention:** Always show `snr_linear` as ratio (e.g., "52.6:1"), NOT dB!

---

### 6. **file_parser.py** (549 lines)
**Purpose:** Parse experimental data files with validation

**Dependencies:**
- re
- numpy
- pandas
- pathlib
- dataclasses
- functools.lru_cache

**Exports:**
- `ParsedFile` dataclass
- `FileParseError` exception
- `FileParser` class

**ParsedFile Properties:**
```python
.compound: str
.file_type: str                 # 'decay' or 'absorption'
.file_path: str
.wavelength: float              # nm
.tau_delta_fixed: float | None
.quantum_yield: float | None    # Present = 'Standard'
.quantum_yield_sd: float | None
.classification: str            # 'Standard' or 'Sample'
.absorbance_at_wavelength: float | List[float]
.data: pd.DataFrame

# Methods:
.get_kinetics_data() -> (time, intensity_replicates)
.get_absorbance_for_replicate(index) -> float
```

**FileParser Methods:**
```python
.parse_directory(directory) -> Dict[compound, List[ParsedFile]]
.parse_file(file_path) -> ParsedFile
.link_absorption_data(parsed_files, directory)
.validate_file_structure(directory) -> Dict[validation_report]
```

**Filename Patterns:**
```python
# Decay: Decay_[Compound]_EX[λ]nm_tauD[value]_QY[value]_QYsd[sd].csv
decay_pattern = r'Decay_(.+?)_EX(\d+(?:\.\d+)?)nm(?:_tauD(\d+(?:\.\d+)?))?(?:_QY(\d+(?:\.\d+)?))?(?:_QYsd(\d+(?:\.\d+)?))?\.csv'

# Absorption: Abs_[Compound].csv
abs_pattern = r'Abs_(.+?)\.csv'
```

**Key Features:**
- Supports dots in compound names (e.g., "Rose.Bengal")
- Auto-detects CSV delimiters (`,`, `\t`, `;`)
- Links absorbance by compound name + wavelength
- Handles 1 to N replicates
- Caches parsed data

---

### 7. **kinetics_analyzer.py** (830 lines)
**Purpose:** Main fitting engine with 3-step workflow

**Dependencies:**
- numpy
- scipy.optimize
- masking_methods
- logger_config
- kinetics_dataclasses
- snr_analyzer

**Note:** Optimized from ~650 → 830 lines (Oct 2025). Now includes vectorized operations and 40-60% performance improvement.

**Exports:**
- `KineticsAnalyzer` class (extends MaskingMethods)

**Key Methods:**
```python
.fit_kinetics(time, intensity, tau_delta_fixed=None) -> KineticsResult
.step1_remove_baseline(time, intensity)
.step2_create_spike_mask(time_experiment, spike_end_experiment)
.step3_fit_both_models(time_experiment, intensity_experiment, fitting_mask, tau_delta_fixed)
.calculate_chi_square(residuals, intensities, n_parameters)
.detect_model_type(fit_result) -> (should_try_single, reasons)
```

**Models:**
```python
# Main model f(t-t0) - PRIMARY (used for QY)
.main_biexponential(t, A, τΔ, τT, t0, y0)
.single_exponential_with_t0(t, A, τΔ, t0, y0)

# Literature model f(t) - COMPARISON ONLY
.literature_biexponential(t, A, τΔ, τT, y0)
.single_exponential(t, A, τΔ, y0)
```

**3-Step Workflow:**
1. **Step 1:** Remove baseline (points before spike_start)
2. **Step 2:** Create spike mask (artifact region)
3. **Step 3:** Fit Main + Literature models with χ² metrics

**Auto Model Selection:**
- Tries biexponential first
- Falls back to single exponential if:
  - τT < 0.05 μs
  - |τΔ - τT| < 0.1 μs
  - R² < 0.85

---

### 8. **statistical_analyzer.py** (653 lines)
**Purpose:** Cross-replicate statistics

**Dependencies:**
- numpy
- kinetics_dataclasses

**Exports:**
- `StatisticalAnalyzer` class

**Key Methods:**
```python
.analyze_replicate_statistics(results: List[KineticsResult]) -> Dict[statistics]
.calculate_means_and_sds(results)
.detect_outliers(results, threshold=2.5)
```

**Output Statistics:**
- Mean ± SD for: A, τΔ, τT, t0, y0, R², χ²ᵣ, SNR
- Outlier detection
- Quality metrics

---

### 9. **quantum_yield_calculator.py** (275 lines)
**Purpose:** QY calculation (ONE method only: A-based)

**Dependencies:**
- numpy
- kinetics_dataclasses

**Exports:**
- `calculate_quantum_yields_simple(kinetics_results, standards, samples) -> (qy_pairs, qy_results)`
- `QuantumYieldCalculator` class (deprecated)

**Formula:**
```python
QY_sample = QY_standard × (A_sample / A_standard) × (Abs_standard / Abs_sample)
```

**Where:**
- A = Amplitude from Main model f(t-t0)
- Abs = Absorbance at excitation wavelength

**Automatic Pairing:**
- Finds Standards (with QY values)
- Finds Samples (without QY)
- Pairs by matching wavelength and solvent

---

### 10. **heterogeneous/ (module package)** (1,837 lines total)
**Purpose:** Heterogeneous systems analysis (vesicles, lipid bilayers)

**Files:**
- `heterogeneous_fitter.py` (313 lines) - Main fitter with presets
- `grid_search.py` (330 lines) - Single-step grid search (NEW: Oct 2025)
- `diffusion_simulator_numba.py` (408 lines) - Numba-JIT diffusion simulation
- `heterogeneous_plotter_new.py` (500 lines) - Chi-square landscape plots (NEW)
- `heterogeneous_dataclasses.py` (285 lines) - Vesicle geometry dataclasses

**Dependencies:**
- numpy, numba
- matplotlib (for plotting)

**Key Features:**
- Single-step grid search with presets (fast/medium/slow)
- τT, τw, τL fitting
- A/B ratio calculation
- Chi-square landscape visualization
- NO quantum yield (not applicable to heterogeneous systems)

**Major Update:** Two-step grid search replaced by single-step algorithm (Oct 2025)

---

### 11. **solis_plotter.py** (1,398 lines)
**Purpose:** Publication-quality plotting (matplotlib-based)

**Dependencies:**
- numpy
- matplotlib
- kinetics_dataclasses

**Note:** Migrated from Plotly → matplotlib (Oct 2025) for better Qt integration

**Exports:**
- `SOLISPlotter` class

**Key Methods:**
```python
.plot_single_decay(result, log_x=True, show_literature=True) -> Figure
.plot_batch_results(results, layout='grid')
.export_pdf(fig, filename)
.export_png(fig, filename, scale=2)
```

**3-Panel Layout:**
1. **Decay curve** (75% height)
   - Black: Experimental data
   - Red: Main fit
   - Orange dashed: Literature fit
   - Vertical line at t0=0
2. **Weighted residuals (literature)**
3. **Weighted residuals (main)**

**Customization:**
- Log/linear scales (X, Y)
- Fonts (family, size)
- Legend position
- Colors (data, fits, residuals)

---

### 12. **csv_exporter.py** (600 lines)
**Purpose:** Export analysis results to CSV

**Dependencies:**
- pandas
- pathlib
- kinetics_dataclasses

**Note:** Enhanced with heterogeneous analysis support (Oct 2025)

**Exports:**
- `CSVExporter` class

**Key Methods:**
```python
.export_kinetics_results(results, filename)
.export_statistics(stats, filename)
.export_quantum_yields(qy_results, filename)
.export_batch_analysis(kinetics, stats, qy, directory)
```

---

### 13. **session_manager.py** (769 lines) ✅ PRODUCTION-READY
**Purpose:** Complete session save/load with dataclass serialization

**Dependencies:**
- json
- pathlib
- numpy
- pandas
- all dataclasses (kinetics_dataclasses, heterogeneous_dataclasses, etc.)

**Status:** ✅ **Complete and production-ready** (Oct 2025)

**Exports:**
- `SessionManager` class
- `NumpyEncoder` class (custom JSON encoder)

**Key Methods:**
```python
.save_session(file_path, session_data) -> bool
.load_session(file_path) -> dict
.serialize_dataclass(obj) -> dict  # Recursive dataclass → dict
.deserialize_dataclass(data, type_name) -> object  # dict → dataclass
```

**Capabilities:**
- ✅ Serialize/deserialize all dataclass types
- ✅ Handle nested dataclasses (KineticsResult contains FitParameters, SNRResult, etc.)
- ✅ Convert NumPy arrays ↔ lists
- ✅ Convert Pandas DataFrames ↔ dicts
- ✅ Preserve type information with `__type__` markers
- ✅ Save browser state (loaded files, analysis results)
- ✅ Save mask corrections
- ⏸️ Plot restoration (user decision pending)

**File Format (.solis.json):**
```json
{
  "version": "1.0",
  "timestamp": "2025-11-02T14:23:45",
  "loaded_compounds": {...},
  "analysis_results": {
    "Compound": [{
      "__type__": "KineticsResult",
      "parameters": {"__type__": "FitParameters", ...},
      "time_experiment_us": {"__ndarray__": [...]},
      ...
    }]
  }
}
```

---

### 14. **file_parser.py** (595 lines)
**Purpose:** CSV parsing with LRU caching

**Dependencies:**
- re, pathlib, functools
- numpy, pandas

**Note:** Enhanced with LRU caching (50-file limit) for performance (Oct 2025)

**Key Features:**
- ✅ Parse decay and absorption files
- ✅ Extract metadata from filenames
- ✅ Link absorbance data to decay files
- ✅ LRU cache for parsed files (performance optimization)
- ✅ Parse excitation intensity (EI) parameter
- ✅ Support dataset type classification

---

## Critical Usage Rules

### 1. **Data Structures**
❌ **NEVER** use dictionaries:
```python
result['A']  # WRONG!
```

✅ **ALWAYS** use dataclasses:
```python
result.parameters.A  # CORRECT
result.fit_quality.r_squared  # CORRECT
result.snr_result.snr_linear  # CORRECT
```

### 2. **SNR Display**
❌ **NEVER** show dB to users:
```python
print(f"SNR: {snr_db:.1f} dB")  # Users don't understand dB!
```

✅ **ALWAYS** show ratio:
```python
print(f"SNR: {snr_linear:.1f}:1")  # "52.6:1" - clear!
```

### 3. **Quantum Yield**
❌ **NO** old methods:
```python
qy_s0_based()  # Doesn't exist!
qy_auc()  # Removed!
```

✅ **ONLY** A-based method:
```python
from quantum_yield_calculator import calculate_quantum_yields_simple
qy_pairs, qy_results = calculate_quantum_yields_simple(kinetics_results)
```

### 4. **Logging**
❌ **NEVER** use print():
```python
print("Analysis complete")  # WRONG!
```

✅ **ALWAYS** use logger:
```python
from logger_config import get_logger
logger = get_logger(__name__)
logger.info("Analysis complete")
```

### 5. **File Parsing**
✅ **Standard workflow:**
```python
from file_parser import FileParser

parser = FileParser()
compounds = parser.parse_directory("path/to/data")

# Link absorption data
for compound, files in compounds.items():
    parser.link_absorption_data(files, "path/to/data")

# Access parsed data
for parsed_file in compounds['Phenalenone']:
    if parsed_file.file_type == 'decay':
        time, replicates = parsed_file.get_kinetics_data()
```

---

## Typical Analysis Workflow

```python
# 1. Parse files
from file_parser import FileParser
parser = FileParser()
compounds = parser.parse_directory("data/")
for compound, files in compounds.items():
    parser.link_absorption_data(files, "data/")

# 2. Analyze kinetics
from kinetics_analyzer import KineticsAnalyzer
analyzer = KineticsAnalyzer()

results = {}
for compound, parsed_files in compounds.items():
    results[compound] = []
    for parsed_file in parsed_files:
        if parsed_file.file_type == 'decay':
            time, replicates = parsed_file.get_kinetics_data()
            for rep_idx, intensity in enumerate(replicates):
                result = analyzer.fit_kinetics(
                    time, intensity,
                    tau_delta_fixed=parsed_file.tau_delta_fixed
                )
                results[compound].append(result)

# 3. Calculate statistics
from statistical_analyzer import StatisticalAnalyzer
stat_analyzer = StatisticalAnalyzer()

stats = {}
for compound, compound_results in results.items():
    stats[compound] = stat_analyzer.analyze_replicate_statistics(compound_results)

# 4. Calculate quantum yields
from quantum_yield_calculator import calculate_quantum_yields_simple
qy_pairs, qy_results = calculate_quantum_yields_simple(results)

# 5. Plot results
from solis_plotter import SOLISPlotter
plotter = SOLISPlotter(output_dir="figures/")

for compound, compound_results in results.items():
    for idx, result in enumerate(compound_results):
        fig = plotter.plot_single_decay(result, log_x=True)
        plotter.export_pdf(fig, f"{compound}_rep{idx+1}.pdf")

# 6. Export results
from csv_exporter import CSVExporter
exporter = CSVExporter()
exporter.export_batch_analysis(results, stats, qy_results, "output/")
```

---

## Notes for GUI Development

### Required from Backend:
1. **File parsing:** `FileParser.parse_directory()` + `link_absorption_data()`
2. **Analysis:** `KineticsAnalyzer.fit_kinetics()` → returns `KineticsResult`
3. **Statistics:** `StatisticalAnalyzer.analyze_replicate_statistics()`
4. **QY calculation:** `calculate_quantum_yields_simple()`
5. **Plotting:** `SOLISPlotter.plot_single_decay()`
6. **Export:** `CSVExporter.export_batch_analysis()`

### Backend Guarantees:
- All functions return **dataclasses**, NOT dictionaries
- SNR is calculated automatically during kinetics fitting
- Model selection (single vs biexponential) is automatic
- χ² metrics are included for quality assessment
- All arrays are numpy arrays
- All errors raise exceptions (no silent failures)

### What Backend Does NOT Do:
- NO GUI widgets
- NO user interaction
- NO file dialogs
- NO progress bars
- NO threading (caller's responsibility)

---

**END OF BACKEND REFERENCE**
