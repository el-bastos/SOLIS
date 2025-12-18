#!/usr/bin/env python3
"""
Test script for session save/load dataclass serialization.

This script creates minimal test dataclasses, saves them, loads them back,
and verifies they are reconstructed as proper dataclass instances (not dicts).

Run this before testing the full SOLIS application to verify the fix works.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.kinetics_dataclasses import KineticsResult, FitParameters, SNRResult, FitQuality, LiteratureModelResult, WorkflowInfo
from utils.session_manager import SessionManager
from utils.logger_config import get_logger

logger = get_logger(__name__)


def create_minimal_kinetics_result():
    """Create a minimal KineticsResult for testing."""

    # Create nested dataclasses
    params = FitParameters(
        A=100.5,
        tau_delta=3.5,
        tau_T=2.0,
        t0=0.1,
        y0=50.0
    )

    fit_quality = FitQuality(
        r_squared=0.98,
        chi_square=1.2,
        reduced_chi_square=1.05,
        residual_standard_error=5.2,
        n_fit_points=500,
        degrees_of_freedom=495,
        model_used='biexponential'
    )

    snr_result = SNRResult(
        snr_db=15.3,
        snr_linear=52.6,
        quality='Good',
        spike_region=None,
        beginning_baseline={'mean': 50.0, 'std': 2.1},
        tail_baseline={'mean': 49.8, 'std': 2.0},
        baseline_used='tail',
        peak_signal=500.0,
        baseline_level=50.0,
        noise_std=2.0
    )

    # Create arrays first (needed for literature model)
    n_points = 500
    time = np.linspace(0, 30, n_points)
    intensity = 100 * np.exp(-time / 3.5) + 50 + np.random.randn(n_points) * 2

    lit_result = LiteratureModelResult(
        success=True,
        A=95.0,
        tau_delta=3.6,
        tau_T=2.1,
        y0=49.5,
        r_squared=0.97,
        chi_square=1.3,
        reduced_chi_square=1.08,
        model_used='biexponential',
        curve=np.linspace(100, 50, n_points),  # Must match n_points
        weighted_residuals=np.random.randn(n_points) * 0.1  # Must match n_points
    )

    workflow = WorkflowInfo(
        method='clean_3_step_workflow',
        baseline_points_removed=10,
        spike_duration=0.5,
        fitting_points=n_points,
        step4_status='deprecated'
    )

    # Create KineticsResult with all nested dataclasses
    result = KineticsResult(
        parameters=params,
        fit_quality=fit_quality,
        time_experiment_us=time,
        intensity_raw=intensity,
        main_curve=100 * np.exp(-time / 3.5) + 50,
        main_weighted_residuals=np.random.randn(n_points) * 0.1,
        literature=lit_result,
        fitting_mask=np.ones(n_points, dtype=bool),
        spike_region=np.zeros(n_points, dtype=bool),
        workflow=workflow,
        snr_result=snr_result
    )

    return result


def test_save_load_cycle():
    """Test complete save/load cycle with dataclass verification."""

    print("=" * 60)
    print("Session Save/Load Dataclass Serialization Test")
    print("=" * 60)

    # Step 1: Create test data
    print("\n[1/5] Creating test KineticsResult with nested dataclasses...")
    result = create_minimal_kinetics_result()
    print(f"  - Created KineticsResult with {len(result.time_experiment_us)} points")
    print(f"  - parameters type: {type(result.parameters).__name__}")
    print(f"  - fit_quality type: {type(result.fit_quality).__name__}")
    print(f"  - snr_result type: {type(result.snr_result).__name__}")
    print(f"  - literature type: {type(result.literature).__name__}")
    print(f"  - workflow type: {type(result.workflow).__name__}")

    # Step 2: Create analysis results structure
    print("\n[2/5] Building analysis results structure...")
    analysis_results = {
        'kinetics_results': {
            'TestCompound': {
                'results': [result],  # List with one result
                'wavelength': 400,
                'classification': 'lag_spike'
            }
        },
        'statistics_results': {},
        'qy_results': {},
        'excluded_count': 0
    }
    print("  - Structure created with 1 compound, 1 replicate")

    # Step 3: Save session
    test_file = Path('test_session.solis.json')
    print(f"\n[3/5] Saving session to {test_file}...")

    success = SessionManager.save_session(
        filepath=test_file,
        analysis_results=analysis_results,
        description="Test session for dataclass serialization verification"
    )

    if not success:
        print("  ERROR: Save failed!")
        return False

    print("  - Save successful")
    print(f"  - File size: {test_file.stat().st_size:,} bytes")

    # Step 4: Load session
    print(f"\n[4/5] Loading session from {test_file}...")

    session_data = SessionManager.load_session(test_file)

    if not session_data:
        print("  ERROR: Load failed!")
        return False

    print("  - Load successful")

    # Step 5: Verify dataclass reconstruction
    print("\n[5/5] Verifying dataclass reconstruction...")

    # Extract loaded result
    loaded_kinetics = session_data['analysis']['homogeneous']['kinetics_results']
    loaded_compound_data = loaded_kinetics['TestCompound']
    loaded_results_list = loaded_compound_data['results']
    loaded_result = loaded_results_list[0]

    # Type checks
    print(f"\n  Loaded result type: {type(loaded_result).__name__}")

    errors = []

    # Check main result
    if not isinstance(loaded_result, KineticsResult):
        errors.append(f"  FAIL: Result is {type(loaded_result).__name__}, not KineticsResult")
    else:
        print("  PASS: Result is KineticsResult instance")

    # Check nested dataclasses
    if not isinstance(loaded_result.parameters, FitParameters):
        errors.append(f"  FAIL: parameters is {type(loaded_result.parameters).__name__}, not FitParameters")
    else:
        print("  PASS: parameters is FitParameters instance")
        print(f"    - A = {loaded_result.parameters.A}")
        print(f"    - tau_delta = {loaded_result.parameters.tau_delta}")

    if not isinstance(loaded_result.fit_quality, FitQuality):
        errors.append(f"  FAIL: fit_quality is {type(loaded_result.fit_quality).__name__}, not FitQuality")
    else:
        print("  PASS: fit_quality is FitQuality instance")
        print(f"    - R² = {loaded_result.fit_quality.r_squared}")

    if not isinstance(loaded_result.snr_result, SNRResult):
        errors.append(f"  FAIL: snr_result is {type(loaded_result.snr_result).__name__}, not SNRResult")
    else:
        print("  PASS: snr_result is SNRResult instance")
        print(f"    - SNR = {loaded_result.snr_result.snr_linear:.1f}:1")

    if not isinstance(loaded_result.literature, LiteratureModelResult):
        errors.append(f"  FAIL: literature is {type(loaded_result.literature).__name__}, not LiteratureModelResult")
    else:
        print("  PASS: literature is LiteratureModelResult instance")

    if not isinstance(loaded_result.workflow, WorkflowInfo):
        errors.append(f"  FAIL: workflow is {type(loaded_result.workflow).__name__}, not WorkflowInfo")
    else:
        print("  PASS: workflow is WorkflowInfo instance")

    # Check NumPy arrays
    if not isinstance(loaded_result.time_experiment_us, np.ndarray):
        errors.append(f"  FAIL: time_experiment_us is {type(loaded_result.time_experiment_us).__name__}, not ndarray")
    else:
        print("  PASS: time_experiment_us is numpy array")
        print(f"    - Shape: {loaded_result.time_experiment_us.shape}")

    # Verify attribute access (this was the original error)
    try:
        _ = loaded_result.parameters.A
        _ = loaded_result.parameters.tau_delta
        print("  PASS: Can access loaded_result.parameters.A (attribute access works)")
    except AttributeError as e:
        errors.append(f"  FAIL: AttributeError on access: {e}")

    # Clean up
    test_file.unlink()
    print(f"\n  - Cleaned up test file: {test_file}")

    # Final result
    print("\n" + "=" * 60)
    if errors:
        print("TEST FAILED - Errors found:")
        for error in errors:
            print(error)
        print("=" * 60)
        return False
    else:
        print("TEST PASSED - All dataclasses reconstructed correctly!")
        print("=" * 60)
        return True


if __name__ == '__main__':
    try:
        success = test_save_load_cycle()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
