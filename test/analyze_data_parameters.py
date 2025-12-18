#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze experimental data file and check parameter compatibility with simulation.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt

# Load data
data_file = r"g:\Meu Drive\Manuscritos\JACS-Au_1O2\SOLIS_CLEAN\examples\heterogeneous\Decay_Pheo_EX400nm_tauD3.5.csv"

print("="*80)
print("ANALYZING EXPERIMENTAL DATA FILE")
print("="*80)
print(f"\nFile: {data_file}")

# Read CSV
time = []
intensity = []

with open(data_file, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 2:
            t = float(parts[0])
            i_str = parts[1]
            time.append(t)
            if i_str:
                intensity.append(float(i_str))
            else:
                intensity.append(np.nan)
        elif len(parts) == 1:
            # Only time, no intensity
            t = float(parts[0])
            time.append(t)
            intensity.append(np.nan)

time = np.array(time)
intensity = np.array(intensity)

print(f"\nTotal data points: {len(time)}")
print(f"Time range: {time[0]:.3f} - {time[-1]:.3f} μs")
print(f"Time step: {np.median(np.diff(time)):.4f} μs")

# Find first valid intensity
first_valid = np.where(~np.isnan(intensity))[0]
if len(first_valid) > 0:
    first_idx = first_valid[0]
    print(f"\nFirst valid data point:")
    print(f"  Time: {time[first_idx]:.3f} μs")
    print(f"  Intensity: {intensity[first_idx]:.1f}")
    print(f"  Index: {first_idx}")

# Remove NaN values for analysis
valid_mask = ~np.isnan(intensity)
time_valid = time[valid_mask]
intensity_valid = intensity[valid_mask]

print(f"\nValid data points: {len(time_valid)}")
print(f"Valid time range: {time_valid[0]:.3f} - {time_valid[-1]:.3f} μs")

# Find peak
peak_idx = np.argmax(intensity_valid)
peak_time = time_valid[peak_idx]
peak_intensity = intensity_valid[peak_idx]

print(f"\nPeak:")
print(f"  Time: {peak_time:.3f} μs")
print(f"  Intensity: {peak_intensity:.1f}")

# Analyze tail (after 20 μs)
tail_mask = time_valid > 20.0
if np.any(tail_mask):
    tail_mean = np.mean(intensity_valid[tail_mask])
    tail_max = np.max(intensity_valid[tail_mask])
    print(f"\nTail (t > 20 μs):")
    print(f"  Mean intensity: {tail_mean:.1f}")
    print(f"  Max intensity: {tail_max:.1f}")
    print(f"  SNR estimate: {peak_intensity/tail_mean:.1f}")

# Check rise time
rise_mask = (time_valid >= 0.3) & (time_valid <= peak_time)
if np.any(rise_mask):
    rise_time = time_valid[rise_mask]
    rise_intensity = intensity_valid[rise_mask]

    # Fit exponential rise: I(t) = I_max * (1 - exp(-t/tau_rise))
    # Simplified: look at time to reach 50% of peak
    half_peak = peak_intensity / 2.0
    half_idx = np.argmin(np.abs(rise_intensity - half_peak))
    t_half = rise_time[half_idx]

    print(f"\nRise characteristics:")
    print(f"  Time to 50% of peak: {t_half:.2f} μs")
    print(f"  This suggests tau_T ≈ {t_half/0.69:.2f} μs (rough estimate)")

# Check decay time (after peak)
decay_mask = (time_valid >= peak_time) & (time_valid <= 30.0)
if np.any(decay_mask):
    decay_time = time_valid[decay_mask]
    decay_intensity = intensity_valid[decay_mask]

    # Simple decay time estimate: time to drop to 1/e of peak
    one_over_e = peak_intensity / np.e
    decay_idx = np.argmin(np.abs(decay_intensity - one_over_e))
    t_decay = decay_time[decay_idx] - peak_time

    print(f"\nDecay characteristics:")
    print(f"  Time from peak to 1/e: {t_decay:.2f} μs")
    print(f"  This suggests tau_eff ≈ {t_decay:.2f} μs")

print("\n" + "="*80)
print("SIMULATION PARAMETER COMPATIBILITY CHECK")
print("="*80)

print("\nDefault simulation parameters:")
print("  Time step: 0.000125 μs (0.125 ns)")
print("  Output step: 0.02 μs (20 ns)")
print("  Max time: 30.0 μs")
print("  Vesicle diameter: 78 nm (layers 1-39)")
print("  Membrane: layers 36-39 (4 nm)")

print("\nData compatibility:")
data_dt = np.median(np.diff(time))
sim_output_dt = 0.02

if np.abs(data_dt - sim_output_dt) < 0.001:
    print(f"  ✓ Time step matches: {data_dt:.4f} μs = {sim_output_dt} μs")
else:
    print(f"  ⚠ Time step mismatch!")
    print(f"    Data: {data_dt:.4f} μs ({data_dt*1000:.1f} ns)")
    print(f"    Simulation: {sim_output_dt} μs ({sim_output_dt*1000:.0f} ns)")

if time_valid[-1] >= 30.0:
    print(f"  ✓ Data extends to {time_valid[-1]:.1f} μs (>= 30 μs)")
else:
    print(f"  ⚠ Data only extends to {time_valid[-1]:.1f} μs (< 30 μs)")

if time_valid[0] <= 0.3:
    print(f"  ✓ Data starts at {time_valid[0]:.2f} μs (<= 0.3 μs)")
else:
    print(f"  ⚠ Data starts late at {time_valid[0]:.2f} μs")

print("\n" + "="*80)
print("POTENTIAL ISSUES")
print("="*80)

issues = []

# Check 1: Is peak time consistent with heterogeneous model?
if peak_time < 1.0:
    issues.append(f"Peak at {peak_time:.2f} μs is very early - might be instrument response")
elif peak_time > 5.0:
    issues.append(f"Peak at {peak_time:.2f} μs is very late - unusual for SUVs")
else:
    print(f"✓ Peak time ({peak_time:.2f} μs) is reasonable for heterogeneous system")

# Check 2: Is there actually a rise phase?
initial_intensity = intensity_valid[0]
if peak_intensity / initial_intensity < 1.5:
    issues.append(f"Weak rise phase (peak/initial = {peak_intensity/initial_intensity:.1f}) - might be homogeneous decay")
else:
    print(f"✓ Clear rise phase (peak/initial = {peak_intensity/initial_intensity:.1f})")

# Check 3: Check for negative values or zeros
if np.any(intensity_valid <= 0):
    n_nonpositive = np.sum(intensity_valid <= 0)
    issues.append(f"{n_nonpositive} data points have intensity <= 0")

# Check 4: Time step consistency
time_diffs = np.diff(time)
if np.std(time_diffs) / np.mean(time_diffs) > 0.01:
    issues.append("Time step is not constant - irregular sampling")

if issues:
    print("\n⚠ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✓ No obvious issues detected")

print("\n" + "="*80)
print("FITTING RECOMMENDATIONS")
print("="*80)

print("\nBased on the data characteristics:")
print(f"  1. Use fit range: {time_valid[0]:.2f} - 30.0 μs")
print(f"  2. Try tau_T range: 1.5 - 3.0 μs (based on rise time)")
print(f"  3. Try tau_Delta_W range: 3.0 - 5.0 μs (based on decay)")
print(f"  4. Start with coarse grid: 10 x 10 points for quick test")

print("\n" + "="*80)
print("SIMULATION vs DATA RESOLUTION")
print("="*80)

print("\nSimulation generates:")
sim_n_points = int(30.0 / 0.02)
print(f"  {sim_n_points} time points (0 to 30 μs in 0.02 μs steps)")

print("\nData has:")
data_n_points_30us = np.sum((time_valid >= 0) & (time_valid <= 30.0))
print(f"  {data_n_points_30us} time points in 0-30 μs range")

if data_n_points_30us == sim_n_points:
    print("  ✓ Perfect match!")
elif data_n_points_30us > sim_n_points:
    print(f"  ℹ Data has MORE points - will be interpolated to simulation grid")
else:
    print(f"  ℹ Data has FEWER points - simulation will be interpolated to data")

# Create diagnostic plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Full signal
ax = axes[0, 0]
ax.plot(time_valid, intensity_valid, 'b.-', markersize=2, linewidth=0.5, alpha=0.7)
ax.axvline(peak_time, color='r', linestyle='--', alpha=0.5, label=f'Peak at {peak_time:.2f} μs')
ax.axvline(0.3, color='g', linestyle='--', alpha=0.5, label='Fit start (0.3 μs)')
ax.set_xlabel('Time (μs)')
ax.set_ylabel('Intensity (counts)')
ax.set_title('Full Signal (0-82 μs)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Fit region (0.3-30 μs)
ax = axes[0, 1]
fit_mask = (time_valid >= 0.3) & (time_valid <= 30.0)
ax.plot(time_valid[fit_mask], intensity_valid[fit_mask], 'b-', linewidth=1.5)
ax.axvline(peak_time, color='r', linestyle='--', alpha=0.5, label=f'Peak')
ax.set_xlabel('Time (μs)')
ax.set_ylabel('Intensity (counts)')
ax.set_title('Fit Region (0.3-30 μs)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Rise phase (0.3 to peak)
ax = axes[1, 0]
rise_mask = (time_valid >= 0.3) & (time_valid <= peak_time + 1.0)
ax.plot(time_valid[rise_mask], intensity_valid[rise_mask], 'g-', linewidth=2)
ax.axhline(peak_intensity/2, color='r', linestyle='--', alpha=0.5, label=f'50% peak')
ax.set_xlabel('Time (μs)')
ax.set_ylabel('Intensity (counts)')
ax.set_title(f'Rise Phase (t_50% ≈ {t_half:.2f} μs)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Log scale to see decay
ax = axes[1, 1]
ax.semilogy(time_valid[fit_mask], intensity_valid[fit_mask], 'b-', linewidth=1.5)
ax.axvline(peak_time, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (μs)')
ax.set_ylabel('Intensity (counts, log scale)')
ax.set_title('Decay Phase (log scale)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = r"g:\Meu Drive\Manuscritos\JACS-Au_1O2\SOLIS_CLEAN\data_analysis.png"
plt.savefig(output_file, dpi=150)
print(f"\n✓ Diagnostic plot saved: {output_file}")

print("\n" + "="*80)
print("DONE")
print("="*80)
