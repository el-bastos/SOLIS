#!/usr/bin/env python3
"""
Numba-Optimized Diffusion Simulator for Singlet Oxygen in SUVs

This is an optimized version using Numba JIT compilation for 10-50x speedup.
Falls back to pure Python if numba is not available.

Based on: Hackbarth & Röder (2015) - ESI Mathematical Formalism
"""

import numpy as np
from typing import Tuple
from heterogeneous.heterogeneous_dataclasses import (
    VesicleGeometry, DiffusionParameters, SimulationResult
)
from utils.logger_config import get_logger

# Try to import numba, fall back gracefully if not available
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define dummy decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = get_logger(__name__)


@jit(nopython=True)
def _diffusion_core_loop(
    n_steps: int,
    steps_per_output: int,
    n_outputs: int,
    dt: float,
    n_layers: int,
    ps_triplet_decay_rate: float,
    ps_layer_indices: np.ndarray,
    membrane_start: int,
    membrane_end: int,
    D_water: float,
    D_lipid: float,
    decay_rates: np.ndarray,
    volumes: np.ndarray,
    partition_coeff: float,
    dx: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core diffusion simulation loop - optimized with Numba JIT.

    CORRECTED IMPLEMENTATION based on Hackbarth & Röder (2015) ESI Equations 6-9.

    Key corrections:
    1. Geometric factors: (j+1)/j for flux j→j+1, j/(j+1) for flux j+1→j (Eq. 9)
    2. Vectorized diffusion updates for all interfaces simultaneously
    3. Partition coefficient correctly applied at phase boundaries
    4. Proper use of D_lipid at phase boundaries per ESI

    Args:
        n_steps: Total number of time steps
        steps_per_output: Steps between outputs
        n_outputs: Number of output time points
        dt: Time step (us)
        n_layers: Number of spatial layers
        ps_triplet_decay_rate: 1/tau_T (1/us)
        ps_layer_indices: Indices where PS generates 1O2
        membrane_start: First membrane layer (1-indexed)
        membrane_end: Last membrane layer (1-indexed)
        D_water: Diffusion coefficient in water (cm^2/s)
        D_lipid: Diffusion coefficient in lipid (cm^2/s)
        decay_rates: 1/tau for each layer (1/us)
        volumes: j^2 for each layer (Eq. 6)
        partition_coeff: Solubility ratio lipid/water (S)
        dx: Layer thickness (cm)

    Returns:
        (n_lipid_output, n_water_output) - time-resolved amounts
    """
    # Output arrays
    n_lipid_output = np.zeros(n_outputs)
    n_water_output = np.zeros(n_outputs)

    # Concentration array [arbitrary units]
    concentration = np.zeros(n_layers)

    # Precompute phase indicators (0-indexed)
    is_lipid = np.zeros(n_layers, dtype=np.bool_)
    for j in range(n_layers):
        layer_idx_1based = j + 1
        is_lipid[j] = (layer_idx_1based >= membrane_start) and (layer_idx_1based <= membrane_end)

    # Precompute geometric coefficients from Equation 9
    # For interface between layer j and j+1 (0-indexed):
    # - Update to layer j: multiply by (j+1)/j where j is 1-indexed = (j_0based+2)/(j_0based+1)
    # - Update to layer j+1: multiply by j/(j+1) where j is 1-indexed = (j_0based+1)/(j_0based+2)
    j_1based = np.arange(1, n_layers + 1, dtype=np.float64)  # 1-indexed layer numbers
    coeff_left = np.zeros(n_layers - 1)   # Coefficient for updating left layer from flux
    coeff_right = np.zeros(n_layers - 1)  # Coefficient for updating right layer from flux

    for i in range(n_layers - 1):
        # Interface i is between layers i and i+1 (0-indexed)
        # In 1-indexed terms: between j=i+1 and j=i+2
        j_left = i + 1   # 1-indexed
        j_right = i + 2  # 1-indexed
        coeff_left[i] = float(j_right) / float(j_left)    # (j+1)/j for left layer
        coeff_right[i] = float(j_left) / float(j_right)   # j/(j+1) for right layer

    # Precompute which interfaces are phase boundaries
    phase_boundary = np.zeros(n_layers - 1, dtype=np.bool_)
    for i in range(n_layers - 1):
        phase_boundary[i] = is_lipid[i] != is_lipid[i+1]

    # Diffusion factors: D·Δt / Δx² (converted to microseconds)
    dt_seconds = dt * 1e-6
    F_water = D_water * dt_seconds / (dx * dx)
    F_lipid = D_lipid * dt_seconds / (dx * dx)

    output_idx = 0

    # Time evolution loop
    for step in range(n_steps):
        current_time = step * dt

        # ============================================
        # STEP 1: Generate ¹O₂ from PS triplet decay
        # ============================================
        # Per ESI: "Generation... is considered before diffusion"
        ps_triplet = np.exp(-current_time * ps_triplet_decay_rate)
        generation_rate = ps_triplet * ps_triplet_decay_rate * dt

        for ps_idx in ps_layer_indices:
            concentration[ps_idx] += generation_rate

        # ============================================
        # STEP 2: Diffusion between layers (Vectorized)
        # ============================================
        # Per ESI Eq. 9: Δc_j = (j+1)/j · D·Δc·Δt/Δx²
        #                Δc_{j+1} = j/(j+1) · D·Δc·Δt/Δx²

        dc_diffusion = np.zeros(n_layers)

        # Process all interfaces simultaneously
        for i in range(n_layers - 1):
            # Interface between layers i and i+1 (0-indexed)
            c_left = concentration[i]
            c_right = concentration[i+1]

            # Apply partition coefficient at phase boundaries (ESI page 2)
            # "the concentration of 1O2 in the lipid layer has to be divided by S"
            if phase_boundary[i]:
                if is_lipid[i] and not is_lipid[i+1]:
                    # Left is lipid, right is water
                    c_left_eff = c_left / partition_coeff
                    c_right_eff = c_right
                else:  # Left is water, right is lipid
                    c_left_eff = c_left
                    c_right_eff = c_right / partition_coeff
            else:
                c_left_eff = c_left
                c_right_eff = c_right

            # Concentration gradient (effective)
            delta_c = c_left_eff - c_right_eff

            # Choose diffusion coefficient
            # Per ESI: "At phase borders... diffusion is calculated based on parameters of lipid"
            if phase_boundary[i]:
                F = F_lipid
            else:
                # Same phase - use appropriate D
                if is_lipid[i]:
                    F = F_lipid
                else:
                    F = F_water

            # Apply Equation 9 updates
            # Flux from left to right: decreases left, increases right
            dc_diffusion[i] -= coeff_left[i] * F * delta_c      # Update left layer
            dc_diffusion[i+1] += coeff_right[i] * F * delta_c   # Update right layer

        # Apply diffusion changes
        concentration += dc_diffusion

        # ============================================
        # STEP 3: Decay of ¹O₂ in each layer
        # ============================================
        # Per ESI: "decay of singlet oxygen... is considered thereafter"
        for j in range(n_layers):
            decay_factor = np.exp(-decay_rates[j] * dt)
            concentration[j] *= decay_factor

        # ============================================
        # STEP 4: Record output at specified intervals
        # ============================================
        if (step + 1) % steps_per_output == 0:
            # Calculate total ¹O₂ in lipid and water
            # Per ESI Eq. 10: n(t) = Σ c_j(t) · j² · V₁
            n_lipid = 0.0
            n_water = 0.0

            for j in range(n_layers):
                amount = concentration[j] * volumes[j]  # volumes[j] = j²

                if is_lipid[j]:
                    n_lipid += amount
                else:
                    n_water += amount

            n_lipid_output[output_idx] = n_lipid
            n_water_output[output_idx] = n_water
            output_idx += 1

    return n_lipid_output, n_water_output


class DiffusionSimulatorNumba:
    """
    Numba-optimized diffusion simulator for ¹O₂ in spherical SUV geometry.

    Uses JIT compilation for 10-50x speedup compared to pure Python.
    """

    def __init__(self, geometry: VesicleGeometry, parameters: DiffusionParameters):
        """
        Initialize simulator with geometry and physical parameters.

        Args:
            geometry: Vesicle geometry (layer structure)
            parameters: Physical parameters (D, tau, etc.)
        """
        self.geometry = geometry
        self.parameters = parameters

        # Pre-calculate layer properties for efficiency
        self._setup_layers()

        if HAS_NUMBA:
            logger.info(f"DiffusionSimulatorNumba initialized (Numba JIT enabled):")
        else:
            logger.warning(f"DiffusionSimulatorNumba initialized (Numba NOT available - using pure Python):")

        logger.info(f"  Layers: {geometry.n_layers}")
        logger.info(f"  Membrane: layers {geometry.membrane_start}-{geometry.membrane_end}")
        logger.info(f"  PS location: layers {geometry.ps_layers}")
        logger.info(f"  Time step: {parameters.time_step} us ({parameters.time_step*1000:.2f} ns)")

    def _setup_layers(self):
        """Pre-calculate layer properties."""
        n = self.geometry.n_layers
        self.dx = self.geometry.layer_thickness * 1e-7  # nm to cm

        # Layer volumes (in units of V₁) - Per ESI Eq. 6: V_j = j² · V₁
        self.volumes = np.arange(1, n + 1, dtype=np.float64) ** 2

        # Decay rate for each layer (1/us)
        self.decay_rates = np.zeros(n)
        for j in range(n):
            layer_idx = j + 1
            is_membrane = self.geometry.is_membrane_layer(layer_idx)
            tau = self.parameters.get_decay_time(is_membrane)
            self.decay_rates[j] = 1.0 / tau

        # PS generation layers (0-indexed)
        self.ps_layer_indices = np.array([l - 1 for l in self.geometry.ps_layers], dtype=np.int64)

    def simulate(self) -> SimulationResult:
        """
        Run diffusion simulation using Numba-optimized core loop.

        Returns:
            SimulationResult with time-resolved n_L(t) and n_W(t)
        """
        n = self.geometry.n_layers
        dt = self.parameters.time_step
        max_time = self.parameters.max_time
        output_dt = self.parameters.output_time_step

        # Number of time steps
        n_steps = int(max_time / dt)
        steps_per_output = int(output_dt / dt)
        n_outputs = int(max_time / output_dt)

        # Output time points
        time_output = np.linspace(0, max_time, n_outputs)

        # PS triplet decay rate
        ps_triplet_decay_rate = 1.0 / self.parameters.tau_T

        logger.info(f"Starting simulation: {n_steps} steps, {n_outputs} outputs")

        # Call the JIT-compiled core loop with corrected parameters
        n_lipid_output, n_water_output = _diffusion_core_loop(
            n_steps=n_steps,
            steps_per_output=steps_per_output,
            n_outputs=n_outputs,
            dt=dt,
            n_layers=n,
            ps_triplet_decay_rate=ps_triplet_decay_rate,
            ps_layer_indices=self.ps_layer_indices,
            membrane_start=self.geometry.membrane_start,
            membrane_end=self.geometry.membrane_end,
            D_water=self.parameters.D_water,
            D_lipid=self.parameters.D_lipid,
            decay_rates=self.decay_rates,
            volumes=self.volumes,
            partition_coeff=self.parameters.partition_coeff,
            dx=self.dx
        )

        logger.info(f"Simulation complete. Peak n_lipid at t={time_output[np.argmax(n_lipid_output)]:.2f} us")

        return SimulationResult(
            time=time_output,
            n_lipid=n_lipid_output,
            n_water=n_water_output,
            tau_T=self.parameters.tau_T,
            tau_delta_water=self.parameters.tau_delta_water,
            geometry=self.geometry,
            parameters=self.parameters
        )

    def get_summary_info(self) -> str:
        """Get summary of simulator configuration."""
        return (
            f"Diffusion Simulator Configuration:\n"
            f"  Numba: {'ENABLED' if HAS_NUMBA else 'DISABLED (pure Python fallback)'}\n"
            f"  Total layers: {self.geometry.n_layers}\n"
            f"  Membrane: layers {self.geometry.membrane_start}-{self.geometry.membrane_end} "
            f"({self.geometry.membrane_thickness} nm)\n"
            f"  PS location: layers {self.geometry.ps_layers}\n"
            f"  Vesicle diameter: {self.geometry.vesicle_diameter:.1f} nm\n"
            f"  D(water): {self.parameters.D_water:.2e} cm2/s\n"
            f"  D(lipid): {self.parameters.D_lipid:.2e} cm2/s\n"
            f"  tau_T: {self.parameters.tau_T:.2f} us\n"
            f"  tau_Delta(water): {self.parameters.tau_delta_water:.2f} us\n"
            f"  tau_Delta(lipid): {self.parameters.tau_delta_lipid:.2f} us\n"
            f"  Partition coeff: {self.parameters.partition_coeff:.1f}\n"
            f"  Time step: {self.parameters.time_step} us ({self.parameters.time_step*1000:.2f} ns)\n"
            f"  Max time: {self.parameters.max_time} us\n"
            f"  Output resolution: {self.parameters.output_time_step} us ({self.parameters.output_time_step*1000:.0f} ns)"
        )


# Quick test and benchmark
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    print("Testing Numba-Optimized DiffusionSimulator...")
    print(f"Numba available: {HAS_NUMBA}")
    print()

    # Create default geometry and parameters
    geometry = VesicleGeometry()
    parameters = DiffusionParameters(tau_T=2.0, tau_delta_water=3.7)

    # Create simulator
    sim = DiffusionSimulatorNumba(geometry, parameters)
    print(sim.get_summary_info())
    print()

    # Benchmark: Run simulation 3 times to see JIT compilation effect
    print("Running benchmark (3 simulations)...")
    times = []

    for i in range(3):
        start = time.time()
        result = sim.simulate()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f} seconds")

    print()
    print(f"First run (with JIT compilation): {times[0]:.2f} s")
    print(f"Subsequent runs (JIT compiled): {times[1]:.2f} s average")
    print(f"Speedup after compilation: {times[0]/times[1]:.1f}x")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Individual components
    ax1.plot(result.time, result.n_lipid, 'b-', label='n_L(t) - Lipid', linewidth=2)
    ax1.plot(result.time, result.n_water, 'r-', label='n_W(t) - Water', linewidth=2)
    ax1.set_xlabel('Time (us)')
    ax1.set_ylabel('Amount of 1O2 (a.u.)')
    ax1.set_title('Simulated 1O2 Kinetics in SUVs (Numba-Optimized)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)

    # Panel 2: Total signal
    rate_ratio = 3.25
    total_signal = rate_ratio * result.n_lipid + result.n_water
    ax2.plot(result.time, total_signal, 'g-', linewidth=2)
    ax2.set_xlabel('Time (us)')
    ax2.set_ylabel('Total Signal (a.u.)')
    ax2.set_title(f'Total Signal (rate ratio = {rate_ratio:.2f})')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)

    plt.tight_layout()
    plt.savefig('test_diffusion_simulator_numba.png', dpi=150)
    print(f"\nTest plot saved: test_diffusion_simulator_numba.png")
    print(f"Peak lipid at t = {result.peak_time_lipid:.2f} us")
    print(f"Peak water at t = {result.peak_time_water:.2f} us")
