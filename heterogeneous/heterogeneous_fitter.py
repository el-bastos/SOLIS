#!/usr/bin/env python3
"""
Heterogeneous Diffusion Model Fitter for Singlet Oxygen in SUVs

Main interface for fitting experimental 1O2 luminescence data using
the heterogeneous diffusion model (Hackbarth & Röder 2015).

Based on:
Hackbarth, S. & Röder, B. (2015)
"Singlet oxygen luminescence kinetics in a heterogeneous environment"
Photochem. Photobiol. Sci., 14, 329-334
"""

import numpy as np
from typing import Tuple, Optional
import time
from heterogeneous.heterogeneous_dataclasses import (
    VesicleGeometry, DiffusionParameters, HeterogeneousFitResult
)
from heterogeneous.diffusion_simulator_numba import DiffusionSimulatorNumba
from heterogeneous.grid_search import GridSearch, GridSearchParams, SimulationCache
from utils.logger_config import get_logger

logger = get_logger(__name__)


class HeterogeneousFitter:
    """
    Main fitter for heterogeneous diffusion model.

    Workflow:
    1. User provides experimental data (time, intensity)
    2. Fitter runs single-step grid search to find best (tau_T, tau_delta_W)
    3. Returns comprehensive fit results with chi2 surface for plotting

    Usage:
        fitter = HeterogeneousFitter()
        result = fitter.fit(time, intensity, preset='medium')
    """

    def __init__(self,
                 geometry: VesicleGeometry = None,
                 base_parameters: DiffusionParameters = None):
        """
        Initialize fitter with default or custom geometry/parameters.

        Args:
            geometry: Vesicle geometry (uses 78nm SUV default if None)
            base_parameters: Physical parameters (uses literature defaults if None)
        """
        self.geometry = geometry or VesicleGeometry()
        self.base_parameters = base_parameters or DiffusionParameters()

        logger.info("HeterogeneousFitter initialized")
        logger.info(f"  Geometry: {self.geometry.n_layers} layers, "
                   f"{self.geometry.vesicle_diameter:.1f} nm diameter")
        logger.info(f"  Membrane: layers {self.geometry.membrane_start}-{self.geometry.membrane_end}")
        logger.info(f"  PS location: layers {self.geometry.ps_layers}")

    def fit(self,
            time_exp: np.ndarray,
            intensity_exp: np.ndarray,
            preset: str = 'medium',
            custom_params: GridSearchParams = None,
            fit_range: Tuple[float, float] = (0.3, 30.0),
            progress_callback=None) -> HeterogeneousFitResult:
        """
        Fit experimental data using heterogeneous diffusion model.

        Args:
            time_exp: Experimental time points (μs)
            intensity_exp: Experimental intensity (photon counts)
            preset: Grid search preset ('fast', 'medium', 'slow')
            custom_params: Custom grid search parameters (overrides preset)
            fit_range: Time range for fitting (μs), default (0.3, 30.0)
            progress_callback: Optional callable(percentage: int) for progress updates

        Returns:
            HeterogeneousFitResult with fitted parameters, curves, and chi2 surface
        """
        start_time = time.time()

        # Select grid search parameters
        if custom_params is not None:
            search_params = custom_params
        elif preset == 'fast':
            search_params = GridSearchParams.fast_preset()
        elif preset == 'slow':
            search_params = GridSearchParams.slow_preset()
        else:  # medium (default)
            search_params = GridSearchParams.medium_preset()

        logger.info(f"Starting fit with preset='{preset}'")
        logger.info(f"  Data: {len(time_exp)} points, time range {time_exp[0]:.2f}-{time_exp[-1]:.2f} us")
        logger.info(f"  Fit range: {fit_range[0]:.2f}-{fit_range[1]:.2f} us")

        # Create grid search
        grid_search = GridSearch(self.geometry, search_params)

        # ============================================
        # STEP 1: Run grid search
        # ============================================
        logger.info("=" * 60)
        logger.info("STEP 1: Running grid search...")
        logger.info("=" * 60)

        tau_T_grid, tau_delta_W_grid, chi2_red_grid = grid_search.run_grid(
            time_exp, intensity_exp, self.base_parameters, fit_range,
            progress_callback=progress_callback
        )

        # Find minimum in grid
        tau_T_best, tau_delta_W_best, chi2_red_best =             grid_search.find_minimum(tau_T_grid, tau_delta_W_grid, chi2_red_grid)

        logger.info(f"Grid minimum: tau_T={tau_T_best:.3f} us, "
                   f"tau_Delta_W={tau_delta_W_best:.3f} us, chi2_red={chi2_red_best:.4f}")

        # ============================================
        # STEP 2: Generate final fit with best parameters
        # ============================================
        logger.info("=" * 60)
        logger.info("STEP 2: Generating final fit curves...")
        logger.info("=" * 60)

        # Run simulation with best parameters
        best_sim_result = grid_search.run_simulation(
            tau_T_best, tau_delta_W_best, self.base_parameters
        )

        # Fit linear model to get amplitudes and final metrics
        best_fit = grid_search.fit_linear_model(
            best_sim_result, time_exp, intensity_exp, fit_range
        )

        # ============================================
        # STEP 3: Prepare output
        # ============================================

        # Calculate weighted residuals for chi-square
        mask = (time_exp >= fit_range[0]) & (time_exp <= fit_range[1])
        weights = 1.0 / np.maximum(intensity_exp[mask], 1.0)
        weighted_residuals = best_fit['residuals'] * np.sqrt(weights)

        # Interpolate simulation curves to full experimental time range
        n_L_full = np.interp(time_exp, best_sim_result.time, best_sim_result.n_lipid)
        n_W_full = np.interp(time_exp, best_sim_result.time, best_sim_result.n_water)

        # Generate full fit curve
        fit_curve_full = best_fit['A'] * n_L_full + best_fit['B'] * n_W_full + best_fit['C']

        # Create result object
        result = HeterogeneousFitResult(
            # Fitted parameters
            tau_T=tau_T_best,
            tau_delta_water=tau_delta_W_best,
            rate_ratio=best_fit['rate_ratio'],
            amplitude_lipid=best_fit['A'],
            amplitude_water=best_fit['B'],
            background=best_fit['C'],

            # Fit quality
            chi_square=best_fit['chi2'],
            reduced_chi_square=best_fit['chi2_red'],
            r_squared=best_fit['r2'],
            degrees_of_freedom=best_fit['dof'],
            n_fit_points=best_fit['n_fit'],

            # Data and curves
            time_experimental=time_exp,
            intensity_experimental=intensity_exp,
            fit_curve=fit_curve_full,
            n_lipid_curve=n_L_full,
            n_water_curve=n_W_full,
            residuals=intensity_exp - fit_curve_full,
            weighted_residuals=weighted_residuals,

            # Simulation info
            geometry=self.geometry,
            parameters=self.base_parameters,

            # Grid search info (for plotting)
            grid_search_surface=chi2_red_grid,
            grid_tau_T=tau_T_grid,
            grid_tau_delta_W=tau_delta_W_grid,
            fit_time_range=fit_range
        )

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("FIT COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"  Total time: {elapsed/60:.1f} minutes")
        logger.info(f"  Cache stats: {grid_search.cache.get_stats()}")
        logger.info(f"\nFinal Results:")
        logger.info(f"  tau_T = {tau_T_best:.3f} us")
        logger.info(f"  tau_Delta(water) = {tau_delta_W_best:.3f} us")
        logger.info(f"  Rate ratio = {best_fit['rate_ratio']:.2f}")
        logger.info(f"  chi2_red = {best_fit['chi2_red']:.4f}")
        logger.info(f"  R^2 = {best_fit['r2']:.4f}")
        logger.info(f"  Quality: {'GOOD' if result.is_good_fit() else 'POOR'}")
        logger.info("=" * 60)

        return result

    def fit_custom_range(self,
                        time_exp: np.ndarray,
                        intensity_exp: np.ndarray,
                        tau_T_range: Tuple[float, float],
                        tau_delta_W_range: Tuple[float, float],
                        grid_points: int = 15,
                        fit_range: Tuple[float, float] = (0.3, 30.0)) -> HeterogeneousFitResult:
        """
        Fit with custom parameter ranges.

        Useful for quick exploration or when you know approximate parameter values.

        Args:
            time_exp: Experimental time points (μs)
            intensity_exp: Experimental intensity (photon counts)
            tau_T_range: (min, max) for tau_T search
            tau_delta_W_range: (min, max) for tau_Delta_W search
            grid_points: Number of points in each dimension (e.g., 15 for 15×15 grid)
            fit_range: Time range for fitting (μs)

        Returns:
            HeterogeneousFitResult
        """
        custom_params = GridSearchParams.custom(
            tau_T_range=tau_T_range,
            tau_delta_W_range=tau_delta_W_range,
            grid_points=grid_points
        )

        return self.fit(time_exp, intensity_exp,
                       custom_params=custom_params,
                       fit_range=fit_range)


# Convenience function for quick fitting
def fit_heterogeneous(time: np.ndarray,
                     intensity: np.ndarray,
                     preset: str = 'medium',
                     fit_range: Tuple[float, float] = (0.3, 30.0)) -> HeterogeneousFitResult:
    """
    Convenience function for quick heterogeneous fitting with defaults.

    Args:
        time: Experimental time points (μs)
        intensity: Experimental intensity (photon counts)
        preset: Grid search preset ('fast', 'medium', 'slow')
        fit_range: Time range for fitting (μs)

    Returns:
        HeterogeneousFitResult

    Example:
        >>> result = fit_heterogeneous(time, intensity, preset='fast')
        >>> print(result)
    """
    fitter = HeterogeneousFitter()
    return fitter.fit(time, intensity, preset=preset, fit_range=fit_range)


# Test with synthetic data
if __name__ == "__main__":
    print("Testing HeterogeneousFitter with synthetic data...")

    # Generate synthetic data (simulate perfect case)
    from heterogeneous.diffusion_simulator_numba import DiffusionSimulatorNumba

    # Generate "experimental" data with known parameters
    geometry = VesicleGeometry()
    params = DiffusionParameters(tau_T=2.0, tau_delta_water=3.7)

    sim = DiffusionSimulatorNumba(geometry, params)
    sim_result = sim.simulate()

    # Create synthetic signal: 3.25 * n_L + n_W + 100 (background)
    rate_ratio_true = 3.25
    signal_true = rate_ratio_true * sim_result.n_lipid + sim_result.n_water + 100.0

    # Add Poisson noise
    np.random.seed(42)
    signal_noisy = np.random.poisson(signal_true)

    print(f"\nSynthetic data generated:")
    print(f"  True tau_T = {params.tau_T:.2f} us")
    print(f"  True tau_Delta_W = {params.tau_delta_water:.2f} us")
    print(f"  True rate ratio = {rate_ratio_true:.2f}")
    print(f"  SNR ~ {np.max(signal_noisy) / np.std(signal_noisy[1000:]):.1f}")

    # Test fitting with fast preset
    print("\n" + "=" * 60)
    print("Testing fit with 'fast' preset...")
    print("=" * 60)

    fitter = HeterogeneousFitter()
    result = fitter.fit(sim_result.time, signal_noisy, preset='fast')

    print("\n" + "=" * 60)
    print("Fit Results:")
    print("=" * 60)
    print(result)

    # Compare with true values
    print("\n" + "=" * 60)
    print("Comparison with True Values:")
    print("=" * 60)
    print(f"  tau_T:        fitted={result.tau_T:.2f} us,  true={params.tau_T:.2f} us,  "
          f"error={abs(result.tau_T - params.tau_T):.2f} us")
    print(f"  tau_Delta_W:  fitted={result.tau_delta_water:.2f} us,  true={params.tau_delta_water:.2f} us,  "
          f"error={abs(result.tau_delta_water - params.tau_delta_water):.2f} us")
    print(f"  Rate ratio:   fitted={result.rate_ratio:.2f},  true={rate_ratio_true:.2f},  "
          f"error={abs(result.rate_ratio - rate_ratio_true):.2f}")

    print("\nTest complete!")
