#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simulation against experimental data to find the issue.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from heterogeneous.heterogeneous_dataclasses import VesicleGeometry, DiffusionParameters, SimulationResult
from heterogeneous.diffusion_simulator_numba import DiffusionSimulatorNumba

print("="*80)
print("TESTING SIMULATION VS EXPERIMENTAL DATA")
print("="*80)

# Load experimental data
data_file = r"g:\Meu Drive\Manuscritos\JACS-Au_1O2\SOLIS_CLEAN\examples\heterogeneous\Decay_Pheo_EX400nm_tauD3.5.csv"

time_exp = []
intensity_exp = []

with open(data_file, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 1:
            t = float(parts[0])
            time_exp.append(t)
            if len(parts) == 2 and parts[1]:
                intensity_exp.append(float(parts[1]))
            else:
                intensity_exp.append(np.nan)

time_exp = np.array(time_exp)
intensity_exp = np.array(intensity_exp)

# Remove NaN
valid = ~np.isnan(intensity_exp)
time_exp = time_exp[valid]
intensity_exp = intensity_exp[valid]

print(f"\nExperimental data:")
print(f"  Points: {len(time_exp)}")
print(f"  Time range: {time_exp[0]:.2f} - {time_exp[-1]:.2f} μs")
print(f"  Peak at: {time_exp[np.argmax(intensity_exp)]:.2f} μs")

# Run simulation with different parameter sets
test_params = [
    {"tau_T": 2.0, "tau_delta_water": 3.7, "label": "Default (2.0, 3.7)"},
    {"tau_T": 2.0, "tau_delta_water": 5.0, "label": "Test (2.0, 5.0)"},
    {"tau_T": 1.5, "tau_delta_water": 3.7, "label": "Fast triplet (1.5, 3.7)"},
    {"tau_T": 2.5, "tau_delta_water": 3.7, "label": "Slow triplet (2.5, 3.7)"},
]

# Create geometry (default)
geometry = VesicleGeometry()

print(f"\nGeometry:")
print(f"  Layers: {geometry.n_layers}")
print(f"  Membrane: {geometry.membrane_start}-{geometry.membrane_end} (4 nm)")
print(f"  PS location: {geometry.ps_layers}")
print(f"  Vesicle diameter: {geometry.vesicle_diameter} nm")

# Run simulations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, param_set in enumerate(test_params):
    print(f"\n{'='*80}")
    print(f"TEST {idx+1}: {param_set['label']}")
    print(f"{'='*80}")

    # Create parameters
    params = DiffusionParameters(
        tau_T=param_set['tau_T'],
        tau_delta_water=param_set['tau_delta_water'],
        max_time=30.0  # Match fitting range
    )

    print(f"  tau_T = {params.tau_T} μs")
    print(f"  tau_Delta_water = {params.tau_delta_water} μs")
    print(f"  tau_Delta_lipid = {params.tau_delta_lipid} μs")
    print(f"  D_water = {params.D_water} cm²/s")
    print(f"  D_lipid = {params.D_lipid} cm²/s")
    print(f"  Partition coeff = {params.partition_coeff}")

    # Run simulation
    simulator = DiffusionSimulatorNumba(geometry, params)
    result = simulator.simulate()

    print(f"  Simulation complete:")
    print(f"    {len(result.time)} time points")
    print(f"    n_lipid peak at {result.time[np.argmax(result.n_lipid)]:.2f} μs")
    print(f"    n_water peak at {result.time[np.argmax(result.n_water)]:.2f} μs")

    # Interpolate simulation to experimental time points
    n_L_interp = np.interp(time_exp, result.time, result.n_lipid)
    n_W_interp = np.interp(time_exp, result.time, result.n_water)

    # Fit linear model: I = A*n_L + B*n_W + C
    # Use fit range 0.3-30 μs
    fit_mask = (time_exp >= 0.3) & (time_exp <= 30.0)
    t_fit = time_exp[fit_mask]
    I_fit = intensity_exp[fit_mask]
    n_L_fit = n_L_interp[fit_mask]
    n_W_fit = n_W_interp[fit_mask]

    # Weighted least squares
    X = np.column_stack([n_L_fit, n_W_fit, np.ones_like(n_L_fit)])
    weights = 1.0 / np.maximum(I_fit, 1.0)
    W = np.diag(weights)

    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ I_fit
    beta = np.linalg.solve(XtWX, XtWy)
    A, B, C = beta

    # Predicted intensity
    I_pred = A * n_L_fit + B * n_W_fit + C

    # Chi-square
    residuals = I_fit - I_pred
    chi2 = np.sum((residuals**2) / np.maximum(I_fit, 1.0))
    dof = len(I_fit) - 3
    chi2_red = chi2 / dof

    # Rate ratio
    rate_ratio = A / B if B > 0 else 0.0

    print(f"  Fit results:")
    print(f"    A = {A:.2e}")
    print(f"    B = {B:.2e}")
    print(f"    C = {C:.2f}")
    print(f"    Rate ratio (A/B) = {rate_ratio:.2f}")
    print(f"    chi2_red = {chi2_red:.3f}")

    # Plot on appropriate subplot
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    ax.plot(t_fit, I_fit, 'b.', markersize=1, alpha=0.3, label='Experimental data')
    ax.plot(t_fit, I_pred, 'r-', linewidth=2, label=f'Fit (χ²_red={chi2_red:.2f})')
    ax.plot(t_fit, A*n_L_fit, 'g--', alpha=0.5, linewidth=1, label=f'A·n_L (lipid)')
    ax.plot(t_fit, B*n_W_fit, 'm--', alpha=0.5, linewidth=1, label=f'B·n_W (water)')

    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Intensity (counts)')
    ax.set_title(f'{param_set["label"]}\nRate ratio={rate_ratio:.2f}, χ²_red={chi2_red:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)

plt.tight_layout()
output_file = r"g:\Meu Drive\Manuscritos\JACS-Au_1O2\SOLIS_CLEAN\simulation_vs_data_test.png"
plt.savefig(output_file, dpi=150)
print(f"\n✓ Test plot saved: {output_file}")

# Additional diagnostic: Check if n_L and n_W have correct shapes
print("\n" + "="*80)
print("DIAGNOSTIC: Simulation Output Analysis")
print("="*80)

# Use default parameters
params_default = DiffusionParameters(tau_T=2.0, tau_delta_water=3.7, max_time=30.0)
sim_default = DiffusionSimulatorNumba(geometry, params_default)
res_default = sim_default.simulate()

print(f"\nSimulation characteristics:")
print(f"  Time points: {len(res_default.time)}")
print(f"  Time range: {res_default.time[0]} - {res_default.time[-1]} μs")

print(f"\nn_lipid:")
print(f"  Min: {np.min(res_default.n_lipid):.3e}")
print(f"  Max: {np.max(res_default.n_lipid):.3e}")
print(f"  Peak time: {res_default.time[np.argmax(res_default.n_lipid)]:.2f} μs")
print(f"  Final value: {res_default.n_lipid[-1]:.3e}")

print(f"\nn_water:")
print(f"  Min: {np.min(res_default.n_water):.3e}")
print(f"  Max: {np.max(res_default.n_water):.3e}")
print(f"  Peak time: {res_default.time[np.argmax(res_default.n_water)]:.2f} μs")
print(f"  Final value: {res_default.n_water[-1]:.3e}")

# Check if ratio makes sense
ratio_peak = res_default.n_lipid / (res_default.n_water + 1e-10)
print(f"\nn_lipid / n_water ratio:")
print(f"  At t=1 μs: {ratio_peak[50]:.2f}")  # Index ~50 is around 1 μs
print(f"  At peak: {ratio_peak[np.argmax(res_default.n_lipid)]:.2f}")
print(f"  At t=10 μs: {ratio_peak[500]:.2f}")  # Index ~500 is around 10 μs

print("\n" + "="*80)
print("CRITICAL CHECKS")
print("="*80)

# Check 1: Do n_L and n_W sum to reasonable total?
total_signal = res_default.n_lipid + res_default.n_water
peak_total = np.max(total_signal)
final_total = total_signal[-1]
print(f"\nTotal ¹O₂ (n_L + n_W):")
print(f"  Peak: {peak_total:.3e}")
print(f"  Final (t=30μs): {final_total:.3e}")
print(f"  Ratio peak/final: {peak_total/final_total:.1f}")

# Check 2: Is lipid component decaying faster than water?
# (It should, because of diffusion out of membrane)
lipid_half = np.where(res_default.n_lipid < np.max(res_default.n_lipid)/2)[0]
water_half = np.where(res_default.n_water < np.max(res_default.n_water)/2)[0]

if len(lipid_half) > 0 and len(water_half) > 0:
    t_lipid_half = res_default.time[lipid_half[0]]
    t_water_half = res_default.time[water_half[0]]
    print(f"\nHalf-life times:")
    print(f"  n_lipid: {t_lipid_half:.2f} μs")
    print(f"  n_water: {t_water_half:.2f} μs")
    if t_lipid_half < t_water_half:
        print(f"  ✓ Lipid decays faster (correct - diffusion out)")
    else:
        print(f"  ⚠ Water decays faster (unexpected!)")

# Check 3: Compare with experimental peak time
exp_peak_time = time_exp[np.argmax(intensity_exp)]
sim_lipid_peak = res_default.time[np.argmax(res_default.n_lipid)]
sim_water_peak = res_default.time[np.argmax(res_default.n_water)]

print(f"\nPeak times comparison:")
print(f"  Experimental data: {exp_peak_time:.2f} μs")
print(f"  Simulation n_lipid: {sim_lipid_peak:.2f} μs")
print(f"  Simulation n_water: {sim_water_peak:.2f} μs")

if abs(exp_peak_time - sim_lipid_peak) < 1.0:
    print(f"  ✓ Close match with lipid component")
elif abs(exp_peak_time - sim_water_peak) < 1.0:
    print(f"  ✓ Close match with water component")
else:
    print(f"  ⚠ Peak time mismatch > 1 μs")

print("\n" + "="*80)
print("DONE")
print("="*80)
