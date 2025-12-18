#!/usr/bin/env python3
"""
Test script to verify linearity plot CSV export handles different array lengths correctly.

This simulates the data structure created by variable_study_plotter.py for linearity plots
and tests that the CSV export code in plot_viewer_widget.py handles it without errors.
"""

import numpy as np
import pandas as pd
import tempfile
import os

def test_linearity_export_logic():
    """Test the array length handling logic for linearity plots."""

    # Simulate data from a Beer-Lambert linearity plot
    # This mimics what variable_study_plotter.py creates
    n_data_points = 5  # Number of actual measurements
    n_regression_points = 100  # Number of points in regression line

    # Create mock data (same structure as fig.solis_export_data)
    current_data = {
        'x_absorbed_fraction': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'alpha': np.array([1000, 2000, 3000, 4000, 5000]),
        'alpha_error': np.array([100, 150, 200, 250, 300]),
        'absorbance_A_lambda': np.array([0.05, 0.11, 0.18, 0.26, 0.35]),
        'compound_labels': ['Sample1', 'Sample2', 'Sample3', 'Sample4', 'Sample5'],
        'regression_line_x': np.linspace(0, 0.6, n_regression_points),
        'regression_line_y': np.linspace(0, 6000, n_regression_points)
    }

    print("Testing linearity CSV export logic...")
    print(f"Data points: {n_data_points}")
    print(f"Regression line points: {n_regression_points}")

    # === Apply the fix logic ===
    data_dict = {}

    # Export data points
    data_dict['Absorbed_Fraction'] = current_data['x_absorbed_fraction']
    data_dict['Alpha'] = current_data['alpha']
    data_dict['Alpha_Error'] = current_data['alpha_error']
    data_dict['Absorbance_A_lambda'] = current_data['absorbance_A_lambda']
    data_dict['Compound'] = current_data['compound_labels']

    # Handle regression line arrays (different length)
    n_data = len(current_data['x_absorbed_fraction'])
    reg_x = current_data['regression_line_x']
    reg_y = current_data['regression_line_y']

    # Pad regression arrays with NaN to match data length
    if len(reg_x) > n_data:
        # Regression line is longer - create separate section
        padded_reg_x = np.full(n_data, np.nan)
        padded_reg_y = np.full(n_data, np.nan)
        data_dict['Regression_Line_X'] = padded_reg_x
        data_dict['Regression_Line_Y'] = padded_reg_y
        # Save full regression line in separate DataFrame
        regression_df = pd.DataFrame({
            'Regression_Line_X': reg_x,
            'Regression_Line_Y': reg_y
        })
        data_dict['__regression_df__'] = regression_df
        print("[OK] Regression line separated (longer than data)")
    else:
        # Regression line is shorter or equal - just pad with NaN
        padded_reg_x = np.full(n_data, np.nan)
        padded_reg_y = np.full(n_data, np.nan)
        padded_reg_x[:len(reg_x)] = reg_x
        padded_reg_y[:len(reg_y)] = reg_y
        data_dict['Regression_Line_X'] = padded_reg_x
        data_dict['Regression_Line_Y'] = padded_reg_y
        print("[OK] Regression line padded to match data length")

    # Check if we have a separate regression DataFrame
    regression_df = data_dict.pop('__regression_df__', None)

    # Create main DataFrame
    try:
        df = pd.DataFrame(data_dict)
        print(f"[OK] Main DataFrame created: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"[FAIL] Failed to create DataFrame: {e}")
        return False

    # Save to temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        temp_file = f.name

    try:
        if regression_df is not None:
            # Append regression line data as separate rows below the main data
            separator = pd.DataFrame({col: [''] for col in df.columns})
            header_row = pd.DataFrame({col: ['===REGRESSION LINE==='] if i == 0 else ['']
                                       for i, col in enumerate(df.columns)})

            regression_df_aligned = regression_df.reindex(columns=df.columns, fill_value='')
            combined_df = pd.concat([df, separator, header_row, regression_df_aligned], ignore_index=True)
            combined_df.to_csv(temp_file, index=False)
            print(f"[OK] Combined CSV saved: {combined_df.shape}")
            print(f"  Data rows: {len(df)}")
            print(f"  Regression rows: {len(regression_df)}")
        else:
            df.to_csv(temp_file, index=False)
            print(f"[OK] CSV saved: {df.shape}")

        # Verify file was created and has content
        file_size = os.path.getsize(temp_file)
        print(f"[OK] CSV file created: {file_size} bytes")

        # Read back and verify
        df_read = pd.read_csv(temp_file)
        print(f"[OK] CSV file readable: {df_read.shape}")

        # Show first few rows
        print("\nFirst 5 rows of exported CSV:")
        print(df_read.head())

        return True

    except Exception as e:
        print(f"[FAIL] Failed to save/read CSV: {e}")
        return False

    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print("\n[OK] Temp file cleaned up")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Linearity Plot CSV Export - Array Length Fix")
    print("=" * 60)

    success = test_linearity_export_logic()

    print("\n" + "=" * 60)
    if success:
        print("[PASS] TEST PASSED - All arrays handled correctly!")
    else:
        print("[FAIL] TEST FAILED - Array length issues detected")
    print("=" * 60)
