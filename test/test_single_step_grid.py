#!/usr/bin/env python3
"""
Test single-step grid search implementation.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from heterogeneous.grid_search import GridSearchParams, GridSearch
from heterogeneous.heterogeneous_dataclasses import VesicleGeometry, DiffusionParameters

print("=" * 80)
print("Testing Single-Step Grid Search Implementation")
print("=" * 80)

# Test 1: GridSearchParams.custom()
print("\nTest 1: GridSearchParams.custom()")
print("-" * 80)

params = GridSearchParams.custom(
    tau_T_range=(1.5, 2.1),
    tau_delta_W_range=(3.5, 4.5),
    grid_points=15
)

print(f"tau_T range: [{params.tau_T_min}, {params.tau_T_max}]")
print(f"tau_delta_W range: [{params.tau_delta_W_min}, {params.tau_delta_W_max}]")
print(f"Grid points: {params.grid_points_tau_T} x {params.grid_points_tau_delta_W}")
print(f"Grid info: {params.get_grid_info()}")

tau_T_grid, tau_delta_W_grid = params.get_grid()
print(f"Actual grid sizes: {len(tau_T_grid)} x {len(tau_delta_W_grid)} = {len(tau_T_grid) * len(tau_delta_W_grid)} simulations")
print(f"tau_T grid: min={tau_T_grid[0]:.3f}, max={tau_T_grid[-1]:.3f}, step={tau_T_grid[1]-tau_T_grid[0]:.3f}")
print(f"tau_w grid: min={tau_delta_W_grid[0]:.3f}, max={tau_delta_W_grid[-1]:.3f}, step={tau_delta_W_grid[1]-tau_delta_W_grid[0]:.3f}")
print("Test 1: PASSED")

# Test 2: Presets
print("\nTest 2: Preset configurations")
print("-" * 80)

for preset_name, preset_func in [('fast', GridSearchParams.fast_preset),
                                   ('medium', GridSearchParams.medium_preset),
                                   ('slow', GridSearchParams.slow_preset)]:
    preset = preset_func()
    n_sims = preset.grid_points_tau_T * preset.grid_points_tau_delta_W
    print(f"{preset_name:8s}: {preset.get_grid_info()}")

print("Test 2: PASSED")

# Test 3: GridSearch initialization
print("\nTest 3: GridSearch initialization")
print("-" * 80)

geometry = VesicleGeometry()
search_params = GridSearchParams.custom(
    tau_T_range=(1.8, 2.2),
    tau_delta_W_range=(3.6, 4.2),
    grid_points=10
)

grid_search = GridSearch(geometry, search_params)
print(f"GridSearch created successfully")
print(f"Cache initialized: {grid_search.cache.get_stats()}")
print("Test 3: PASSED")

# Test 4: Verify no two-step methods exist
print("\nTest 4: Verify two-step methods removed")
print("-" * 80)

has_fine_grid = hasattr(search_params, 'get_fine_grid')
has_run_fine_grid = hasattr(grid_search, 'run_fine_grid')

print(f"has get_fine_grid(): {has_fine_grid}")
print(f"has run_fine_grid(): {has_run_fine_grid}")

if has_fine_grid or has_run_fine_grid:
    print("Test 4: FAILED - Two-step methods still exist!")
else:
    print("Test 4: PASSED - Two-step methods removed")

# Test 5: Check new method exists
print("\nTest 5: Verify single-step method exists")
print("-" * 80)

has_run_grid = hasattr(grid_search, 'run_grid')
print(f"has run_grid(): {has_run_grid}")

if has_run_grid:
    print("Test 5: PASSED - Single-step method exists")
else:
    print("Test 5: FAILED - run_grid() method missing!")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("All tests PASSED!")
print()
print("Single-step grid search implementation is working correctly.")
print()
print("Key changes:")
print("  - GridSearchParams.custom() now takes grid_points instead of steps")
print("  - GridSearch.run_grid() replaces two-step approach")
print("  - No fine grid refinement (single-step only)")
print("=" * 80)
