#!/usr/bin/env python3
"""
Minimal test demonstrating the DataFrame dtype bug with absorption files.

This test shows the exact failure mode:
- Absorption files have integer column names: [0, 1, 2, 3]
- After JSON serialization/deserialization, columns become object dtype
- NumPy operations fail with TypeError
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


def test_integer_column_names():
    """Minimal reproduction of the dtype bug with integer column names."""

    print("\n" + "="*70)
    print("TEST: DataFrame with Integer Column Names")
    print("="*70)

    # Create DataFrame similar to absorption files
    print("\n1. Creating DataFrame with integer column names (like absorption files):")
    test_df = pd.DataFrame({
        0: np.array([300.0, 310.0, 320.0, 330.0, 340.0], dtype=np.float64),
        1: np.array([0.10, 0.15, 0.20, 0.18, 0.12], dtype=np.float64),
        2: np.array([0.11, 0.16, 0.21, 0.19, 0.13], dtype=np.float64),
        3: np.array([0.09, 0.14, 0.19, 0.17, 0.11], dtype=np.float64)
    })

    print(f"   Columns: {list(test_df.columns)}")
    print(f"   Column types: {[type(c) for c in test_df.columns]}")
    print("\n   Original dtypes:")
    for col, dtype in test_df.dtypes.items():
        print(f"      {col}: {dtype}")

    # Verify they're float64
    assert all(test_df[col].dtype == np.float64 for col in test_df.columns)
    print("   [OK] All columns are float64")

    # Create ParsedFile (absorption type)
    parsed_file = ParsedFile(
        compound='TestCompound',
        file_type='absorption',
        file_path='test/Abs_Test.csv',
        wavelength=None,
        tau_delta_fixed=None,
        quantum_yield=None,
        quantum_yield_sd=None,
        excitation_intensity=None,
        intensity_unit=None,
        classification='Standard',
        absorbance_at_wavelength=None,
        dataset_type=None,
        data=test_df
    )

    loaded_compounds = {'TestCompound': [parsed_file]}

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir) / 'test_absorption.solis.json'

        print(f"\n2. Saving session...")
        SessionManager.save_session(
            filepath=session_path,
            loaded_compounds=loaded_compounds,
            description="Test absorption file dtype bug"
        )
        print("   [OK] Session saved")

        print("\n3. Loading session...")
        loaded_session = SessionManager.load_session(session_path)
        print("   [OK] Session loaded")

        # Get restored DataFrame
        restored_file = loaded_session['data']['loaded_compounds']['TestCompound'][0]
        df_restored = restored_file.data

        print("\n4. Checking restored DataFrame:")
        print(f"   Columns: {list(df_restored.columns)}")
        print(f"   Column types: {[type(c) for c in df_restored.columns]}")

        print("\n   Restored dtypes:")
        for col in df_restored.columns:
            dtype = df_restored[col].dtype
            values_type = type(df_restored[col].values)
            first_elem_type = type(df_restored[col].values[0]) if len(df_restored) > 0 else None

            print(f"      Column {col}:")
            print(f"         dtype: {dtype}")
            print(f"         values type: {values_type}")
            print(f"         first element type: {first_elem_type}")

            if dtype == object:
                print(f"         [FAIL] Object dtype detected!")
                # Check if it's a list
                first_val = df_restored[col].iloc[0]
                print(f"         first value: {first_val} (type: {type(first_val)})")

        # Now try NumPy operations
        print("\n5. Testing NumPy operations:")

        try:
            print("   Attempting np.isnan() on column 1...")
            col_data = df_restored[1]
            print(f"      Column dtype: {col_data.dtype}")
            print(f"      Column values type: {type(col_data.values)}")

            # This will fail if dtype is object
            nan_mask = np.isnan(col_data)
            print("      [OK] np.isnan() succeeded")

        except TypeError as e:
            print(f"      [FAIL] TypeError: {e}")
            print("\n   ⚠️ DTYPE BUG CONFIRMED!")
            print("   Integer column names cause DataFrame dtype corruption.")
            return False

        print("\n6. Trying array operations...")

        try:
            # Mean
            mean_val = np.mean(df_restored[1].values)
            print(f"   Mean of column 1: {mean_val:.4f}")
            print("   [OK] np.mean() succeeded")

            # Diff
            diff_val = np.diff(df_restored[0].values)
            print(f"   Diff of column 0: {diff_val}")
            print("   [OK] np.diff() succeeded")

        except TypeError as e:
            print(f"   [FAIL] TypeError in array operations: {e}")
            return False

    print("\n" + "="*70)
    print("[UNEXPECTED] Test passed - dtype bug not reproduced!")
    print("="*70)
    print("\nThis suggests the bug may be specific to:")
    print("  - How file_parser.py creates DataFrames")
    print("  - CSV parsing edge cases")
    print("  - Specific pandas version behavior")
    print("="*70 + "\n")

    return True


def test_string_column_names():
    """Control test - string column names should work."""

    print("\n" + "="*70)
    print("TEST: DataFrame with String Column Names (Control)")
    print("="*70)

    # Create DataFrame with string column names
    test_df = pd.DataFrame({
        'wavelength': np.array([300.0, 310.0, 320.0], dtype=np.float64),
        'abs_1': np.array([0.10, 0.15, 0.20], dtype=np.float64)
    })

    print(f"\n   Original dtypes: {dict(test_df.dtypes)}")

    parsed_file = ParsedFile(
        compound='Control',
        file_type='absorption',
        file_path='test/Abs_Control.csv',
        wavelength=None,
        tau_delta_fixed=None,
        quantum_yield=None,
        quantum_yield_sd=None,
        excitation_intensity=None,
        intensity_unit=None,
        classification='Standard',
        absorbance_at_wavelength=None,
        dataset_type=None,
        data=test_df
    )

    loaded_compounds = {'Control': [parsed_file]}

    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir) / 'control.solis.json'

        SessionManager.save_session(
            filepath=session_path,
            loaded_compounds=loaded_compounds,
            description="Control test"
        )

        loaded_session = SessionManager.load_session(session_path)
        restored_file = loaded_session['data']['loaded_compounds']['Control'][0]
        df_restored = restored_file.data

        print(f"   Restored dtypes: {dict(df_restored.dtypes)}")

        # Test NumPy operations
        try:
            np.isnan(df_restored['abs_1'])
            mean = np.mean(df_restored['wavelength'].values)
            print(f"   Mean wavelength: {mean:.1f}")
            print("   [OK] All operations successful")
            return True
        except TypeError as e:
            print(f"   [FAIL] Unexpected error: {e}")
            return False


if __name__ == '__main__':
    print("\n" + "#"*70)
    print("# Absorption File DataFrame Dtype Bug Test")
    print("#"*70)

    # Run control test first
    control_passed = test_string_column_names()

    # Run problem test
    problem_passed = test_integer_column_names()

    print("\n" + "#"*70)
    print("# RESULTS")
    print("#"*70)
    print(f"Control Test (string columns): {'PASSED' if control_passed else 'FAILED'}")
    print(f"Problem Test (integer columns): {'PASSED' if problem_passed else 'FAILED'}")

    if not problem_passed:
        print("\n[CONFIRMED] DataFrame dtype bug present with integer column names")
        print("Solution: Implement pickle-based serialization")
        sys.exit(1)
    else:
        print("\n[UNEXPECTED] Bug not reproduced in this test")
        print("Need to investigate further with actual file_parser.py output")
        sys.exit(0)
