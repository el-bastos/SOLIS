#!/usr/bin/env python3
"""
Test to verify the asdict() bug fix in session_manager.py

This test ensures that ParsedFile dataclasses are properly serialized
and deserialized with type markers preserved.
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


def test_parsed_file_serialization():
    """Test that ParsedFile objects serialize and deserialize correctly."""

    print("\n" + "="*70)
    print("TEST: ParsedFile Serialization/Deserialization")
    print("="*70)

    # Create a test ParsedFile with DataFrame
    test_df = pd.DataFrame({
        'time': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        'intensity_0': np.array([100.0, 150.0, 120.0, 90.0, 85.0]),
        'intensity_1': np.array([105.0, 155.0, 125.0, 95.0, 88.0])
    })

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

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir) / 'test_session.solis.json'

        print(f"\n1. Saving session to: {session_path}")
        success = SessionManager.save_session(
            filepath=session_path,
            loaded_compounds=loaded_compounds,
            description="Test session for bug fix verification"
        )

        if not success:
            print("   [FAIL] FAILED: Session save failed")
            return False

        print("   [OK] Session saved successfully")

        # Check file size
        file_size_mb = session_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {file_size_mb:.2f} MB")

        # Load the session
        print("\n2. Loading session...")
        loaded_session = SessionManager.load_session(session_path)

        if loaded_session is None:
            print("   [FAIL] FAILED: Session load returned None")
            return False

        print("   [OK] Session loaded successfully")

        # Verify loaded_compounds structure
        print("\n3. Verifying loaded data...")

        if 'data' not in loaded_session:
            print("   [FAIL] FAILED: 'data' key missing from loaded session")
            return False

        if 'loaded_compounds' not in loaded_session['data']:
            print("   [FAIL] FAILED: 'loaded_compounds' missing from data")
            return False

        restored_compounds = loaded_session['data']['loaded_compounds']

        if restored_compounds is None:
            print("   [FAIL] FAILED: loaded_compounds is None")
            return False

        if 'TestCompound' not in restored_compounds:
            print("   [FAIL] FAILED: TestCompound not in restored data")
            print(f"   Available keys: {list(restored_compounds.keys())}")
            return False

        restored_files = restored_compounds['TestCompound']

        if not isinstance(restored_files, list) or len(restored_files) == 0:
            print("   [FAIL] FAILED: restored_files is not a list or is empty")
            print(f"   Type: {type(restored_files)}")
            return False

        restored_parsed_file = restored_files[0]

        # CRITICAL TEST: Check if it's a ParsedFile instance, not a dict
        print(f"\n4. Type checking (CRITICAL):")
        print(f"   Original type: {type(test_parsed_file)}")
        print(f"   Restored type: {type(restored_parsed_file)}")

        if not isinstance(restored_parsed_file, ParsedFile):
            print(f"   [FAIL] FAILED: Restored object is {type(restored_parsed_file)}, not ParsedFile!")
            print("   This means the asdict() bug is still present.")
            return False

        print("   [OK] Object is ParsedFile instance (bug is FIXED!)")

        # Test attribute access (this would crash if it were a dict)
        print("\n5. Testing attribute access:")
        try:
            compound_name = restored_parsed_file.compound
            wavelength = restored_parsed_file.wavelength
            qy = restored_parsed_file.quantum_yield
            df = restored_parsed_file.data

            print(f"   compound: {compound_name}")
            print(f"   wavelength: {wavelength}")
            print(f"   quantum_yield: {qy}")
            print(f"   data type: {type(df)}")
            print(f"   data shape: {df.shape if df is not None else 'None'}")

            # Verify values match
            assert compound_name == 'TestCompound', "Compound name mismatch"
            assert wavelength == 400.0, "Wavelength mismatch"
            assert qy == 0.98, "Quantum yield mismatch"
            assert isinstance(df, pd.DataFrame), "DataFrame not restored"
            assert df.shape == (5, 3), f"DataFrame shape mismatch: {df.shape}"

            print("   [OK] All attributes accessible and correct")

        except AttributeError as e:
            print(f"   [FAIL] FAILED: AttributeError - {e}")
            print("   This means deserialization returned a dict instead of dataclass")
            return False
        except Exception as e:
            print(f"   [FAIL] FAILED: Unexpected error - {e}")
            return False

        # Test DataFrame contents
        print("\n6. Verifying DataFrame contents:")
        if restored_parsed_file.data is not None:
            df_restored = restored_parsed_file.data

            # Check columns
            expected_cols = ['time', 'intensity_0', 'intensity_1']
            if list(df_restored.columns) != expected_cols:
                print(f"   [FAIL] Column mismatch: {list(df_restored.columns)} vs {expected_cols}")
                return False

            # Check data values
            if not np.allclose(df_restored['time'].values, test_df['time'].values):
                print("   [FAIL] Time data mismatch")
                return False

            print("   [OK] DataFrame contents match original")
        else:
            print("   [WARN]  WARNING: DataFrame is None after deserialization")

    print("\n" + "="*70)
    print("[OK] ALL TESTS PASSED - Bug fix verified!")
    print("="*70 + "\n")
    return True


if __name__ == '__main__':
    success = test_parsed_file_serialization()
    sys.exit(0 if success else 1)
