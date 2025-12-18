#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test if two-step grid search misses global minimum vs single dense grid.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from heterogeneous.heterogeneous_fitter import HeterogeneousFitter
from heterogeneous.grid_search import GridSearchParams
from heterogeneous.heterogeneous_dataclasses import VesicleGeometry, DiffusionParameters

print("="*80)
print("TESTING: Two-Step Grid Search vs Single Dense Grid")
print("="*80)
print("\nHypothesis: Two-step approach gets stuck in local minimum")
print("            Single dense grid finds global minimum")
print()

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

print(f"Experimental data loaded:")
print(f"  {len(time_exp)} points, {time_exp[0]:.2f} - {time_exp[-1]:.2f} μs")
print(f"  Peak at {time_exp[np.argmax(intensity_exp)]:.2f} μs")
print()

# Create fitter
geometry = VesicleGeometry()
base_params = DiffusionParameters()
fitter = HeterogeneousFitter(geometry, base_params)

# ============================================================================
# TEST 1: Current two-step approach (medium preset)
# ============================================================================
print("="*80)
print("TEST 1: Two-Step Grid Search (Current Default)")
print("="*80)
print("  Coarse: 1.0-3.5 μs in 0.1 steps, 3.0-5.0 μs in 0.1 steps")
print("  Fine: ±0.2 μs around coarse minimum in 0.02 steps")
print()

twostep_params = GridSearchParams.medium_preset()
print(f"Running two-step grid search...")
print(f"  Estimated: ~{26*21 + 11*11} simulations, ~{(26*21 + 11*11)*1.2/60:.1f} minutes")
print()

result_twostep = fitter.fit(
    time_exp, intensity_exp,
    custom_params=twostep_params,
    fit_range=(0.3, 30.0)
)

print(f"\nTwo-Step Result:")
print(f"  tau_T = {result_twostep.tau_T:.3f} μs")
print(f"  tau_Delta_W = {result_twostep.tau_delta_water:.3f} μs")
print(f"  chi2_red = {result_twostep.reduced_chi_square:.4f}")
print(f"  Rate ratio = {result_twostep.rate_ratio:.2f}")
print(f"  R^2 = {result_twostep.r_squared:.4f}")

# ============================================================================
# TEST 2: Single dense grid (like Hackbarth & Röder paper)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Single Dense Grid (Paper Method)")
print("="*80)
print("  Range: 1.0-3.5 μs, 3.0-5.0 μs")
print("  Step: 0.05 μs (compromise between speed and resolution)")
print()

# Use 0.05 instead of 0.02 for speed (still much denser than 0.1!)
single_params = GridSearchParams.custom(
    tau_T_range=(1.0, 3.5),
    tau_T_step=0.05,
    tau_delta_W_range=(3.0, 5.0),
    tau_delta_W_step=0.05,
    fine_step=0.0,  # No fine grid
    fine_radius=0.0
)

n_tau_T = int((3.5 - 1.0) / 0.05) + 1
n_tau_W = int((5.0 - 3.0) / 0.05) + 1
print(f"Running single dense grid...")
print(f"  {n_tau_T} × {n_tau_W} = {n_tau_T * n_tau_W} simulations")
print(f"  Estimated time: ~{n_tau_T * n_tau_W * 1.2 / 60:.1f} minutes")
print()

result_single = fitter.fit(
    time_exp, intensity_exp,
    custom_params=single_params,
    fit_range=(0.3, 30.0)
)

print(f"\nSingle Dense Grid Result:")
print(f"  tau_T = {result_single.tau_T:.3f} μs")
print(f"  tau_Delta_W = {result_single.tau_delta_water:.3f} μs")
print(f"  chi2_red = {result_single.reduced_chi_square:.4f}")
print(f"  Rate ratio = {result_single.rate_ratio:.2f}")
print(f"  R^2 = {result_single.r_squared:.4f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

delta_tau_T = abs(result_single.tau_T - result_twostep.tau_T)
delta_tau_W = abs(result_single.tau_delta_water - result_twostep.tau_delta_water)
delta_chi2 = result_single.reduced_chi_square - result_twostep.reduced_chi_square

print(f"\nParameter Differences:")
print(f"  Δtau_T = {delta_tau_T:.3f} μs")
print(f"  Δtau_Delta_W = {delta_tau_W:.3f} μs")
print(f"  Δchi2_red = {delta_chi2:.4f}")

print(f"\nChi-Square Comparison:")
print(f"  Two-step: {result_twostep.reduced_chi_square:.4f}")
print(f"  Single:   {result_single.reduced_chi_square:.4f}")

if delta_chi2 < -0.01:
    print(f"\n✓ SINGLE GRID IS BETTER by {abs(delta_chi2):.4f}")
    print("  → Two-step grid likely found LOCAL minimum")
    print("  → BUG CONFIRMED!")
elif delta_chi2 > 0.01:
    print(f"\n⚠ TWO-STEP IS BETTER by {delta_chi2:.4f}")
    print("  → This is unexpected! May indicate other issues")
else:
    print(f"\n≈ BOTH METHODS GIVE SAME RESULT")
    print("  → Two-step grid appears to work correctly for this data")

if delta_tau_T > 0.1 or delta_tau_W > 0.1:
    print(f"\n⚠ SIGNIFICANT PARAMETER DIFFERENCE")
    print(f"  → Methods found different minima!")
    print(f"  → Chi-square landscape has multiple local minima")
else:
    print(f"\n✓ Parameters agree within 0.1 μs")

# ============================================================================
# VISUALIZE CHI-SQUARE LANDSCAPES
# ============================================================================
print("\n" + "="*80)
print("Generating chi-square landscape plots...")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Two-step coarse grid
ax = axes[0]
tau_T_grid = result_twostep.grid_tau_T
tau_W_grid = result_twostep.grid_tau_delta_W
chi2_grid = result_twostep.grid_search_surface

# Create meshgrid for contour plot
TT, TW = np.meshgrid(tau_T_grid, tau_W_grid, indexing='ij')

# Plot contours
levels = np.linspace(np.min(chi2_grid), np.min(chi2_grid) * 3, 20)
contour = ax.contour(TT, TW, chi2_grid, levels=levels, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# Mark two-step minimum
ax.plot(result_twostep.tau_T, result_twostep.tau_delta_water,
        'ro', markersize=15, label=f'Two-step min\n({result_twostep.tau_T:.2f}, {result_twostep.tau_delta_water:.2f})')

# Mark single grid minimum
ax.plot(result_single.tau_T, result_single.tau_delta_water,
        'b^', markersize=15, label=f'Single grid min\n({result_single.tau_T:.2f}, {result_single.tau_delta_water:.2f})')

ax.set_xlabel('tau_T (μs)', fontsize=12)
ax.set_ylabel('tau_Delta_W (μs)', fontsize=12)
ax.set_title('Two-Step Coarse Grid Chi-Square Landscape', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Single grid (if we stored it)
ax = axes[1]
# For single grid, we need to extract from result_single
# Since it used no fine grid, the grid_search_surface should be the full grid
if hasattr(result_single, 'grid_search_surface'):
    tau_T_grid_s = result_single.grid_tau_T
    tau_W_grid_s = result_single.grid_tau_delta_W
    chi2_grid_s = result_single.grid_search_surface

    TT_s, TW_s = np.meshgrid(tau_T_grid_s, tau_W_grid_s, indexing='ij')

    levels_s = np.linspace(np.min(chi2_grid_s), np.min(chi2_grid_s) * 3, 20)
    contour_s = ax.contour(TT_s, TW_s, chi2_grid_s, levels=levels_s, cmap='viridis')
    ax.clabel(contour_s, inline=True, fontsize=8)

    ax.plot(result_single.tau_T, result_single.tau_delta_water,
            'b^', markersize=15, label=f'Minimum\n({result_single.tau_T:.2f}, {result_single.tau_delta_water:.2f})')

    ax.set_xlabel('tau_T (μs)', fontsize=12)
    ax.set_ylabel('tau_Delta_W (μs)', fontsize=12)
    ax.set_title('Single Dense Grid Chi-Square Landscape', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = r"g:\Meu Drive\Manuscritos\JACS-Au_1O2\SOLIS_CLEAN\grid_search_comparison.png"
plt.savefig(output_file, dpi=150)
print(f"\n✓ Plot saved: {output_file}")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if delta_chi2 < -0.05:
    print("\n🚨 BUG CONFIRMED: Two-step grid search found LOCAL minimum")
    print(f"  Two-step chi2_red: {result_twostep.reduced_chi_square:.4f}")
    print(f"  Single chi2_red:   {result_single.reduced_chi_square:.4f}")
    print(f"  Improvement: {abs(delta_chi2):.4f} ({abs(delta_chi2)/result_twostep.reduced_chi_square*100:.1f}%)")
    print()
    print("RECOMMENDATION:")
    print("  1. Use single dense grid (0.05 or 0.02 μs steps)")
    print("  2. OR: Decrease coarse step to 0.05 μs")
    print("  3. OR: Increase fine grid radius to 0.5 μs")
elif abs(delta_chi2) < 0.01 and (delta_tau_T < 0.05 and delta_tau_W < 0.05):
    print("\n✓ Two-step grid search appears to work correctly for this data")
    print("  Both methods found the same minimum")
    print()
    print("CONCLUSION:")
    print("  The fitting problem must be elsewhere:")
    print("  - Data quality issues?")
    print("  - Model doesn't fit data?")
    print("  - User expectations wrong?")
else:
    print("\n⚠ INCONCLUSIVE")
    print("  Methods gave different results, but neither is clearly better")
    print("  Need to examine chi-square landscapes manually")

print("\n" + "="*80)
print("DONE")
print("="*80)
