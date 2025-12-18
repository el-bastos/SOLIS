#!/usr/bin/env python3
"""
Test to verify DataFrame dtype preservation in session save/load.

This test specifically checks for the DataFrame column dtype bug:
- DataFrames serialized as JSON → columns become Python lists
- Deserialized DataFrames have object dtype (lists), not NumPy arrays
- NumPy operations fail: np.isnan(list) → crash

This test will FAIL if the dtype issue is still present.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.session_manager import SessionManager
from data.file_parser import ParsedFile


def test_dataframe_dtype_preservation():
    """Test that DataFrame dtypes are preserved through serialization."""

    print("\n" + "="*70)
    print("TEST: DataFrame Dtype Preservation")
    print("="*70)

    # Create a test DataFrame with specific dtypes
    test_df = pd.DataFrame({
        'time': np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        'intensity_0': np.array([100.0, 150.0, 120.0, 90.0, 85.0], dtype=np.float64),
        'intensity_1': np.array([105.0, 155.0, np.nan, 95.0, 88.0], dtype=np.float64)  # Include NaN
    })

    print("\n1. Original DataFrame dtypes:")
    for col, dtype in test_df.dtypes.items():
        print(f"   {col}: {dtype}")

    # Verify original dtypes are correct
    assert test_df['time'].dtype == np.float64
    assert test_df['intensity_0'].dtype == np.float64
    assert test_df['intensity_1'].dtype == np.float64
    print("   [OK] Original dtypes are float64")

    # Create a test ParsedFile
    test_parsed_file = ParsedFile(
        compound='TestCompound',
        file_type='decay',
        file_path='test/data/test_file.csv',
        wavelength=400.0,
        tau_delta_fixed=3.5,
        quantum_yield=0.98,
        quantum_yield_sd=0.08,
        excitation_intensity=10.0,
        intensity_unit='mW',
        classification='Standard',
        absorbance_at_wavelength=0.5,
        dataset_type='lag_spike',
        data=test_df
    )

    # Create a mock loaded_compounds structure
    loaded_compounds = {
        'TestCompound': [test_parsed_file]
    }

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir) / 'test_dtype_session.solis.json'

        print(f"\n2. Saving session to: {session_path.name}")
        success = SessionManager.save_session(
            filepath=session_path,
            loaded_compounds=loaded_compounds,
            description="Test session for dtype preservation"
        )

        if not success:
            print("   [FAIL] Session save failed")
            return False

        print("   [OK] Session saved successfully")

        # Load the session
        print("\n3. Loading session...")
        loaded_session = SessionManager.load_session(session_path)

        if loaded_session is None:
            print("   [FAIL] Session load returned None")
            return False

        print("   [OK] Session loaded successfully")

        # Get the restored DataFrame
        restored_compounds = loaded_session['data']['loaded_compounds']
        restored_parsed_file = restored_compounds['TestCompound'][0]
        df_restored = restored_parsed_file.data

        # CRITICAL TEST 1: Check dtypes
        print("\n4. Checking restored DataFrame dtypes:")
        print("   EXPECTED: All columns should be float64 (NumPy arrays)")
        print("   ACTUAL:")

        all_correct = True
        for col in df_restored.columns:
            dtype = df_restored[col].dtype
            print(f"   {col}: {dtype}")

            if dtype != np.float64:
                print(f"      [FAIL] Expected float64, got {dtype}")
                all_correct = False

            # Check if it's actually a NumPy array or a Python list
            first_val = df_restored[col].iloc[0]
            print(f"      Type of values: {type(df_restored[col].values)}")
            print(f"      Type of first value: {type(first_val)}")

            if not isinstance(df_restored[col].values, np.ndarray):
                print(f"      [FAIL] Column is not backed by NumPy array!")
                all_correct = False

        if not all_correct:
            print("\n   [FAIL] DataFrame dtypes are NOT preserved correctly")
            print("   This is the bug mentioned in Session 43!")
            return False
        else:
            print("\n   [OK] All dtypes preserved correctly")

        # CRITICAL TEST 2: Try NumPy operations (this is what fails in real usage)
        print("\n5. Testing NumPy operations on restored data:")

        try:
            # Test 1: np.isnan() - This is what crashes in real code
            print("   Testing np.isnan() on intensity_1 column (has NaN)...")
            nan_mask = np.isnan(df_restored['intensity_1'])
            nan_count = np.sum(nan_mask)
            print(f"      Found {nan_count} NaN value(s)")
            print("      [OK] np.isnan() works")

            # Test 2: Direct array operations
            print("   Testing array arithmetic (mean calculation)...")
            mean_time = np.mean(df_restored['time'].values)
            print(f"      Mean time: {mean_time:.2f}")
            print("      [OK] Array arithmetic works")

            # Test 3: Filtering with boolean masks
            print("   Testing boolean indexing...")
            filtered = df_restored[df_restored['time'] > 1.5]
            print(f"      Filtered {len(filtered)} rows where time > 1.5")
            print("      [OK] Boolean indexing works")

            # Test 4: Direct NumPy operations that fail with lists
            print("   Testing np.diff() operation...")
            time_diff = np.diff(df_restored['time'].values)
            print(f"      Time differences: {time_diff}")
            print("      [OK] np.diff() works")

        except TypeError as e:
            print(f"\n   [FAIL] NumPy operation FAILED with TypeError:")
            print(f"   {e}")
            print("\n   This confirms the DataFrame dtype bug!")
            print("   Columns are Python lists, not NumPy arrays.")
            return False

        except Exception as e:
            print(f"\n   [FAIL] Unexpected error during NumPy operations:")
            print(f"   {e}")
            return False

        print("\n   [OK] All NumPy operations successful")

        # CRITICAL TEST 3: Check actual column values type
        print("\n6. Deep inspection of column data structure:")

        for col in ['time', 'intensity_0', 'intensity_1']:
            print(f"\n   Column: {col}")
            series = df_restored[col]
            values = series.values

            print(f"      Series dtype: {series.dtype}")
            print(f"      Values type: {type(values)}")
            print(f"      Values dtype: {values.dtype}")
            print(f"      First value type: {type(values[0])}")

            # Check if values is actually an ndarray or object array of lists
            if isinstance(values, np.ndarray):
                if values.dtype == object:
                    print(f"      [WARN] NumPy array with object dtype!")
                    # Check if elements are lists
                    if isinstance(values[0], list):
                        print(f"      [FAIL] Array contains lists, not numbers!")
                        print(f"      This is the dtype bug - JSON deserialization issue")
                        return False
                    else:
                        print(f"      Elements are {type(values[0])}, not lists")
                else:
                    print(f"      [OK] Native NumPy array with numeric dtype")
            else:
                print(f"      [FAIL] Not a NumPy array!")
                return False

    print("\n" + "="*70)
    print("[OK] ALL DTYPE TESTS PASSED!")
    print("="*70)
    print("\nDataFrame serialization preserves dtypes correctly.")
    print("NumPy operations work on restored data.")
    print("The dtype bug mentioned in Session 43 is NOT present.")
    print("="*70 + "\n")

    return True


def test_real_world_scenario():
    """Test a real-world scenario that would fail with the dtype bug."""

    print("\n" + "="*70)
    print("TEST: Real-World Analysis Scenario")
    print("="*70)

    # Simulate what kinetics_analyzer.py does
    print("\n1. Simulating analysis pipeline on restored data...")

    # Create test data similar to real SOLIS data
    test_df = pd.DataFrame({
        'time': np.linspace(0, 100, 1000, dtype=np.float64),
        'intensity_0': 1000 * np.exp(-np.linspace(0, 100, 1000) / 25.0) + 5 * np.random.randn(1000)
    })

    test_parsed_file = ParsedFile(
        compound='RealTest',
        file_type='decay',
        file_path='test/data/real_test.csv',
        wavelength=532.0,
        tau_delta_fixed=None,
        quantum_yield=None,
        quantum_yield_sd=None,
        excitation_intensity=50.0,
        intensity_unit='mW',
        classification='Sample',
        absorbance_at_wavelength=None,
        dataset_type='lag_spike',
        data=test_df
    )

    loaded_compounds = {'RealTest': [test_parsed_file]}

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir) / 'real_world_test.solis.json'

        SessionManager.save_session(
            filepath=session_path,
            loaded_compounds=loaded_compounds,
            description="Real-world test"
        )

        loaded_session = SessionManager.load_session(session_path)
        restored_file = loaded_session['data']['loaded_compounds']['RealTest'][0]
        df = restored_file.data

        # Now try operations that kinetics_analyzer.py would do
        print("   Simulating spike detection algorithm...")

        try:
            # Extract time and intensity
            time = df['time'].values
            intensity = df['intensity_0'].values

            # Operations from spike_detector_last.py
            print("      - Computing differences (np.diff)")
            intensity_diff = np.diff(intensity)

            print("      - Computing rolling statistics")
            rolling_mean = np.convolve(intensity, np.ones(10)/10, mode='valid')

            print("      - Detecting outliers with boolean masks")
            threshold = np.mean(intensity) + 3 * np.std(intensity)
            outliers = intensity > threshold
            n_outliers = np.sum(outliers)

            print(f"      - Found {n_outliers} outliers")

            print("      - Masking NaN values")
            valid_mask = ~np.isnan(intensity)
            valid_count = np.sum(valid_mask)

            print(f"      - {valid_count}/{len(intensity)} valid points")

            # Operations from core_fitting.py
            print("      - Computing Poisson weights")
            weights = 1.0 / np.sqrt(np.abs(intensity) + 1)

            print("      - Calculating residuals")
            fitted = intensity  # Simplified
            residuals = intensity - fitted
            weighted_residuals = residuals * weights

            print("\n   [OK] All analysis operations successful!")
            print("   Restored data is fully functional for analysis.")

        except TypeError as e:
            print(f"\n   [FAIL] Analysis pipeline failed with TypeError:")
            print(f"   {e}")
            print("\n   This would break real usage of SOLIS after loading a session!")
            return False

        except Exception as e:
            print(f"\n   [FAIL] Analysis pipeline failed:")
            print(f"   {e}")
            return False

    print("\n" + "="*70)
    print("[OK] REAL-WORLD SCENARIO PASSED!")
    print("="*70 + "\n")

    return True


if __name__ == '__main__':
    print("\n" + "#"*70)
    print("# DataFrame Dtype Preservation Test Suite")
    print("#"*70)

    test1_passed = test_dataframe_dtype_preservation()
    test2_passed = test_real_world_scenario()

    print("\n" + "#"*70)
    print("# FINAL RESULTS")
    print("#"*70)
    print(f"Test 1 (Dtype Preservation): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Real-World Scenario): {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("\n[OK] ALL TESTS PASSED!")
        print("Session save/load is working correctly for DataFrames.")
        sys.exit(0)
    else:
        print("\n[FAIL] SOME TESTS FAILED!")
        print("The DataFrame dtype bug is still present.")
        print("See Session 43 notes for the pickle-based solution.")
        sys.exit(1)
