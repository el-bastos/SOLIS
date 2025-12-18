#!/usr/bin/env python3
"""
Compare v2 heterogeneous implementation against current implementation.
Find where they diverge to identify the bug.
"""

import numpy as np
import sys

# Test v2 implementation
sys.path.insert(0, 'hetpast/v2')
from diffusion_simulator_numba import DiffusionSimulatorNumba as SimulatorV2
from heterogeneous_dataclasses import VesicleGeometry as GeometryV2, DiffusionParameters as ParametersV2

# Test current implementation
sys.path.insert(0, '.')
from heterogeneous.diffusion_simulator_numba import DiffusionSimulatorNumba as SimulatorCurrent
from heterogeneous.heterogeneous_dataclasses import VesicleGeometry as GeometryCurrent, DiffusionParameters as ParametersCurrent

print("="*70)
print("COMPARING V2 vs CURRENT HETEROGENEOUS IMPLEMENTATIONS")
print("="*70)

# Create identical parameters
tau_T = 2.0
tau_delta_water = 3.7

# V2 setup
geometry_v2 = GeometryV2()
params_v2 = ParametersV2(tau_T=tau_T, tau_delta_water=tau_delta_water)
sim_v2 = SimulatorV2(geometry_v2, params_v2)

# Current setup
geometry_current = GeometryCurrent()
params_current = ParametersCurrent(tau_T=tau_T, tau_delta_water=tau_delta_water)
sim_current = SimulatorCurrent(geometry_current, params_current)

print(f"\nParameters:")
print(f"  tau_T = {tau_T} us")
print(f"  tau_delta_water = {tau_delta_water} us")
print(f"  tau_delta_lipid = {params_v2.tau_delta_lipid} us")
print(f"  D_water = {params_v2.D_water} cm^2/s")
print(f"  D_lipid = {params_v2.D_lipid} cm^2/s")
print(f"  Partition coeff = {params_v2.partition_coeff}")
print(f"  Geometry: {geometry_v2.n_layers} layers, membrane {geometry_v2.membrane_start}-{geometry_v2.membrane_end}")

print(f"\nRunning simulations...")

# Run v2
result_v2 = sim_v2.simulate()
print(f"  V2: Complete")

# Run current
result_current = sim_current.simulate()
print(f"  Current: Complete")

print(f"\nComparing results...")

# Compare time arrays
time_match = np.allclose(result_v2.time, result_current.time)
print(f"  Time arrays match: {time_match}")
if not time_match:
    print(f"    V2 time: {result_v2.time[:5]}")
    print(f"    Current time: {result_current.time[:5]}")

# Compare n_lipid
n_lipid_match = np.allclose(result_v2.n_lipid, result_current.n_lipid, rtol=1e-3)
print(f"  n_lipid arrays match (rtol=1e-3): {n_lipid_match}")
if not n_lipid_match:
    max_diff = np.max(np.abs(result_v2.n_lipid - result_current.n_lipid))
    max_rel_diff = np.max(np.abs(result_v2.n_lipid - result_current.n_lipid) / (np.abs(result_v2.n_lipid) + 1e-10))
    print(f"    Max absolute difference: {max_diff:.6e}")
    print(f"    Max relative difference: {max_rel_diff:.6e}")

    # Find where they first diverge significantly
    rel_diff = np.abs(result_v2.n_lipid - result_current.n_lipid) / (np.abs(result_v2.n_lipid) + 1e-10)
    first_diverge = np.where(rel_diff > 0.01)[0]
    if len(first_diverge) > 0:
        idx = first_diverge[0]
        print(f"\n    First significant divergence at index {idx} (t={result_v2.time[idx]:.3f} us):")
        print(f"      V2 n_lipid: {result_v2.n_lipid[idx]:.6e}")
        print(f"      Current n_lipid: {result_current.n_lipid[idx]:.6e}")
        print(f"      Rel diff: {rel_diff[idx]:.3%}")

# Compare n_water
n_water_match = np.allclose(result_v2.n_water, result_current.n_water, rtol=1e-3)
print(f"  n_water arrays match (rtol=1e-3): {n_water_match}")
if not n_water_match:
    max_diff = np.max(np.abs(result_v2.n_water - result_current.n_water))
    max_rel_diff = np.max(np.abs(result_v2.n_water - result_current.n_water) / (np.abs(result_v2.n_water) + 1e-10))
    print(f"    Max absolute difference: {max_diff:.6e}")
    print(f"    Max relative difference: {max_rel_diff:.6e}")

# Compare peaks
v2_peak_time = result_v2.time[np.argmax(result_v2.n_lipid)]
current_peak_time = result_current.time[np.argmax(result_current.n_lipid)]
v2_peak_val = np.max(result_v2.n_lipid)
current_peak_val = np.max(result_current.n_lipid)

print(f"\n  Peak analysis (n_lipid):")
print(f"    V2: t={v2_peak_time:.3f} μs, value={v2_peak_val:.6e}")
print(f"    Current: t={current_peak_time:.3f} μs, value={current_peak_val:.6e}")
print(f"    Peak time difference: {abs(v2_peak_time - current_peak_time):.3f} μs")
print(f"    Peak value ratio: {current_peak_val / v2_peak_val:.6f}")

# Compare total signal (rate_ratio * n_L + n_W)
rate_ratio = 3.25
signal_v2 = rate_ratio * result_v2.n_lipid + result_v2.n_water
signal_current = rate_ratio * result_current.n_lipid + result_current.n_water

signal_match = np.allclose(signal_v2, signal_current, rtol=1e-3)
print(f"\n  Total signal (3.25*n_L + n_W) match: {signal_match}")

if signal_match:
    print("\n✅ IMPLEMENTATIONS ARE EQUIVALENT!")
else:
    print("\n❌ IMPLEMENTATIONS DIFFER!")
    print("\nThis explains why the current implementation gives wrong fit results!")

print("\n" + "="*70)
