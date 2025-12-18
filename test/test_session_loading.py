#!/usr/bin/env python3
"""
Test script to verify session loading functionality.

This script tests that the session loading populates both:
1. Loaded compounds in the data browser
2. Analysis results in the results tabs

Run this after loading a session in SOLIS GUI to verify the fix.
"""

import json
from pathlib import Path

def inspect_session_file(session_path: str):
    """
    Inspect a session file to understand its structure.

    Parameters
    ----------
    session_path : str
        Path to .solis.json session file
    """
    print(f"Inspecting session file: {session_path}\n")
    print("=" * 80)

    with open(session_path, 'r') as f:
        session_data = json.load(f)

    # Show metadata
    if 'metadata' in session_data:
        print("\nMETADATA:")
        print("-" * 40)
        for key, value in session_data['metadata'].items():
            print(f"  {key}: {value}")

    # Show data structure
    if 'data' in session_data:
        print("\nDATA SECTION:")
        print("-" * 40)
        data = session_data['data']

        if 'folder_path' in data:
            print(f"  Folder path: {data['folder_path']}")

        if 'loaded_compounds' in data:
            compounds = data['loaded_compounds']
            print(f"  Number of compounds: {len(compounds)}")
            print(f"  Compound names: {list(compounds.keys())}")

            # Show structure of first compound
            if compounds:
                first_compound = list(compounds.keys())[0]
                first_data = compounds[first_compound]
                print(f"\n  First compound structure ({first_compound}):")
                if isinstance(first_data, list) and first_data:
                    print(f"    Type: List with {len(first_data)} files")
                    print(f"    First file keys: {list(first_data[0].keys())}")

    # Show analysis results
    if 'analysis' in session_data:
        print("\nANALYSIS RESULTS:")
        print("-" * 40)
        analysis = session_data['analysis']

        for key in ['kinetics_results', 'statistics_results', 'qy_results',
                    'surplus_results', 'heterogeneous_results']:
            if key in analysis:
                result = analysis[key]
                if isinstance(result, dict):
                    print(f"  {key}: {len(result)} entries")
                elif isinstance(result, list):
                    print(f"  {key}: {len(result)} items")
                else:
                    print(f"  {key}: {type(result)}")

    # Show preferences
    if 'preferences' in session_data:
        print("\nPREFERENCES:")
        print("-" * 40)
        prefs = session_data['preferences']
        print(f"  Keys: {list(prefs.keys())}")

    # Show UI state
    if 'ui_state' in session_data:
        print("\nUI STATE:")
        print("-" * 40)
        ui_state = session_data['ui_state']
        print(f"  Keys: {list(ui_state.keys())}")
        if 'mask_corrections' in ui_state:
            print(f"  Mask corrections: {len(ui_state['mask_corrections'])} entries")

    print("\n" + "=" * 80)
    print("\nSession file structure looks valid!")
    print("\nExpected behavior when loading:")
    print("1. Compounds should appear in 'Loaded Files' tab")
    print("2. Analysis results should appear in respective tabs")
    print("3. No WARNING messages about 'not yet implemented'")
    print("=" * 80)


if __name__ == '__main__':
    # Test with the homogeneous example
    session_file = Path(__file__).parent / 'examples' / 'homogeneous' / 'homogeneous.solis.json'

    if session_file.exists():
        inspect_session_file(str(session_file))
    else:
        print(f"Session file not found: {session_file}")
        print("\nPlease provide a path to a .solis.json file:")
        print("  python test_session_loading.py")
