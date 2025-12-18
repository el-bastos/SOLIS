#!/usr/bin/env python3
"""
Single-Step Grid Search for Heterogeneous Diffusion Model Fitting

Strategy:
Single dense grid over user-defined parameter ranges.
Grid density controlled by number of points (e.g., 15×15 like datagen6).

Example: 15×15 grid over tau_T (1.5-2.1), tau_w (3.5-4.5)
   → 225 simulations, ~4-5 minutes with Numba

Based on: Hackbarth & Röder (2015), Fig. 4
Proven approach: datagen6 results (chi2_red=0.9365 with 14×14 grid)
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
import time
from heterogeneous.heterogeneous_dataclasses import (
    VesicleGeometry, DiffusionParameters, SimulationResult
)
from heterogeneous.diffusion_simulator_numba import DiffusionSimulatorNumba
from utils.logger_config import get_logger

logger = get_logger(__name__)


@dataclass
class GridSearchParams:
    """Parameters for single-step grid search."""
    # Grid ranges
    tau_T_min: float = 1.0
    tau_T_max: float = 3.5
    tau_delta_W_min: float = 3.0
    tau_delta_W_max: float = 5.0

    # Grid density (number of points in each dimension)
    grid_points_tau_T: int = 15
    grid_points_tau_delta_W: int = 15

    @classmethod
    def fast_preset(cls):
        """
        Fast preset: Coarse grid for quick exploration.
        ~100 simulations, ~2 minutes with Numba
        """
        return cls(
            tau_T_min=1.0, tau_T_max=3.5,
            tau_delta_W_min=3.0, tau_delta_W_max=5.0,
            grid_points_tau_T=10,
            grid_points_tau_delta_W=10
        )

    @classmethod
    def medium_preset(cls):
        """
        Medium preset: Standard grid (like datagen6).
        ~225 simulations, ~5 minutes with Numba
        """
        return cls(
            tau_T_min=1.0, tau_T_max=3.5,
            tau_delta_W_min=3.0, tau_delta_W_max=5.0,
            grid_points_tau_T=15,
            grid_points_tau_delta_W=15
        )

    @classmethod
    def slow_preset(cls):
        """
        Slow preset: Dense grid for high precision.
        ~625 simulations, ~12 minutes with Numba
        """
        return cls(
            tau_T_min=1.0, tau_T_max=3.5,
            tau_delta_W_min=3.0, tau_delta_W_max=5.0,
            grid_points_tau_T=25,
            grid_points_tau_delta_W=25
        )

    @classmethod
    def custom(cls, tau_T_range, tau_delta_W_range, grid_points):
        """
        Custom preset: User-defined ranges and grid density.

        Args:
            tau_T_range: (min, max) for tau_T in us
            tau_delta_W_range: (min, max) for tau_Delta_W in us
            grid_points: Number of points in each dimension (e.g., 15 for 15×15 grid)

        Example:
            >>> params = GridSearchParams.custom(
            ...     tau_T_range=(1.5, 2.1),
            ...     tau_delta_W_range=(3.5, 4.5),
            ...     grid_points=15
            ... )
        """
        return cls(
            tau_T_min=tau_T_range[0], tau_T_max=tau_T_range[1],
            tau_delta_W_min=tau_delta_W_range[0], tau_delta_W_max=tau_delta_W_range[1],
            grid_points_tau_T=grid_points,
            grid_points_tau_delta_W=grid_points
        )

    def get_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get grid points for τ_T and τ_Δ,W.

        Returns:
            (tau_T_grid, tau_delta_W_grid)
        """
        tau_T_grid = np.linspace(self.tau_T_min, self.tau_T_max, self.grid_points_tau_T)
        tau_delta_W_grid = np.linspace(self.tau_delta_W_min, self.tau_delta_W_max,
                                        self.grid_points_tau_delta_W)
        return tau_T_grid, tau_delta_W_grid

    def get_grid_info(self) -> str:
        """Get human-readable grid information."""
        tau_T_step = (self.tau_T_max - self.tau_T_min) / (self.grid_points_tau_T - 1) if self.grid_points_tau_T > 1 else 0
        tau_w_step = (self.tau_delta_W_max - self.tau_delta_W_min) / (self.grid_points_tau_delta_W - 1) if self.grid_points_tau_delta_W > 1 else 0
        n_total = self.grid_points_tau_T * self.grid_points_tau_delta_W
        return (f"{self.grid_points_tau_T}×{self.grid_points_tau_delta_W} = {n_total} simulations, "
                f"tau_T step={tau_T_step:.3f}us, tau_w step={tau_w_step:.3f}us")


class SimulationCache:
    """Cache for simulation results to avoid recomputation."""

    def __init__(self):
        self.cache: Dict[Tuple[float, float], SimulationResult] = {}
        self.hits = 0
        self.misses = 0

    def get(self, tau_T: float, tau_delta_W: float):
        """Get cached simulation result or None."""
        key = (round(tau_T, 3), round(tau_delta_W, 3))
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, tau_T: float, tau_delta_W: float, result: SimulationResult):
        """Store simulation result in cache."""
        key = (round(tau_T, 3), round(tau_delta_W, 3))
        self.cache[key] = result

    def get_stats(self) -> str:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = 100.0 * self.hits / total if total > 0 else 0.0
        return f"Cache: {len(self.cache)} entries, {self.hits}/{total} hits ({hit_rate:.1f}%)"


class GridSearch:
    """
    Single-step grid search for heterogeneous model fitting.

    Workflow:
    1. Run grid of simulations over parameter space
    2. For each grid point, fit experimental data and calculate χ²
    3. Find minimum χ² in grid
    """

    def __init__(self, geometry: VesicleGeometry,
                 search_params: GridSearchParams = None):
        """
        Initialize grid search.

        Args:
            geometry: Vesicle geometry
            search_params: Grid search parameters (uses defaults if None)
        """
        self.geometry = geometry
        self.search_params = search_params or GridSearchParams()
        self.cache = SimulationCache()

        logger.info("GridSearch initialized (single-step)")
        logger.info(f"  tau_T range: [{self.search_params.tau_T_min}, {self.search_params.tau_T_max}] us")
        logger.info(f"  tau_Delta_W range: [{self.search_params.tau_delta_W_min}, {self.search_params.tau_delta_W_max}] us")
        logger.info(f"  Grid: {self.search_params.get_grid_info()}")

    def run_simulation(self, tau_T: float, tau_delta_W: float,
                      base_params: DiffusionParameters) -> SimulationResult:
        """Run simulation for given parameters, using cache if available."""
        # Check cache first
        cached = self.cache.get(tau_T, tau_delta_W)
        if cached is not None:
            return cached

        # Run simulation
        params = DiffusionParameters(
            D_water=base_params.D_water,
            D_lipid=base_params.D_lipid,
            tau_delta_lipid=base_params.tau_delta_lipid,
            tau_T=tau_T,
            tau_delta_water=tau_delta_W,
            partition_coeff=base_params.partition_coeff,
            time_step=base_params.time_step,
            max_time=base_params.max_time,
            output_time_step=base_params.output_time_step
        )

        simulator = DiffusionSimulatorNumba(self.geometry, params)
        result = simulator.simulate()

        # Cache result
        self.cache.put(tau_T, tau_delta_W, result)

        return result

    def fit_linear_model(self, sim_result: SimulationResult,
                        time_exp: np.ndarray, intensity_exp: np.ndarray,
                        fit_range: Tuple[float, float] = (0.3, 30.0)) -> Dict:
        """Fit linear model: Signal = A·n_L(t) + B·n_W(t) + C"""
        # Interpolate simulation to experimental time points
        n_L_interp = np.interp(time_exp, sim_result.time, sim_result.n_lipid)
        n_W_interp = np.interp(time_exp, sim_result.time, sim_result.n_water)

        # Apply fit range mask
        mask = (time_exp >= fit_range[0]) & (time_exp <= fit_range[1])
        t_fit = time_exp[mask]
        y_fit = intensity_exp[mask]
        n_L_fit = n_L_interp[mask]
        n_W_fit = n_W_interp[mask]

        # Build design matrix for linear least squares: y = X·β
        # X = [n_L, n_W, 1], β = [A, B, C]^T
        X = np.column_stack([n_L_fit, n_W_fit, np.ones_like(n_L_fit)])

        # Weighted least squares (weights = 1/intensity for Poisson)
        weights = 1.0 / np.maximum(y_fit, 1.0)
        W = np.diag(weights)

        # Solve: β = (X^T W X)^{-1} X^T W y
        try:
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y_fit
            beta = np.linalg.solve(XtWX, XtWy)
            A, B, C = beta
        except np.linalg.LinAlgError:
            logger.warning(f"Singular matrix in linear fit for tau_T={sim_result.tau_T:.2f}, "
                          f"tau_Delta_W={sim_result.tau_delta_water:.2f}")
            return {
                'A': 0.0, 'B': 0.0, 'C': 0.0,
                'chi2': 1e10, 'chi2_red': 1e10, 'r2': -1.0,
                'rate_ratio': 0.0, 'n_fit': len(y_fit), 'dof': 0
            }

        # Calculate fit curve
        y_pred = A * n_L_fit + B * n_W_fit + C
        residuals = y_fit - y_pred

        # Chi-square (Poisson statistics)
        chi2_terms = (residuals ** 2) / np.maximum(y_fit, 1.0)
        chi2 = np.sum(chi2_terms)

        # Degrees of freedom
        dof = len(y_fit) - 3
        chi2_red = chi2 / dof if dof > 0 else np.inf

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Rate ratio
        rate_ratio = A / B if B > 0 else 0.0

        return {
            'A': A, 'B': B, 'C': C,
            'chi2': chi2, 'chi2_red': chi2_red, 'r2': r2,
            'rate_ratio': rate_ratio, 'n_fit': len(y_fit),
            'dof': dof,
            'y_pred': y_pred,
            'residuals': residuals
        }

    def run_grid(self, time_exp: np.ndarray, intensity_exp: np.ndarray,
                 base_params: DiffusionParameters,
                 fit_range: Tuple[float, float] = (0.3, 30.0),
                 progress_callback=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run grid search. Returns (tau_T_grid, tau_delta_W_grid, chi2_red_grid)

        Args:
            progress_callback: Optional callable(percentage: int) called every ~10 simulations
        """
        tau_T_grid, tau_delta_W_grid = self.search_params.get_grid()
        n_tau_T = len(tau_T_grid)
        n_tau_delta_W = len(tau_delta_W_grid)
        chi2_red_grid = np.full((n_tau_T, n_tau_delta_W), np.inf)

        logger.info(f"Running grid search: {n_tau_T} x {n_tau_delta_W} = {n_tau_T * n_tau_delta_W} simulations")
        start_time = time.time()

        for i, tau_T in enumerate(tau_T_grid):
            for j, tau_delta_W in enumerate(tau_delta_W_grid):
                # Run simulation
                sim_result = self.run_simulation(tau_T, tau_delta_W, base_params)

                # Fit linear model
                fit_result = self.fit_linear_model(sim_result, time_exp, intensity_exp, fit_range)

                chi2_red_grid[i, j] = fit_result['chi2_red']

                n_done = i * n_tau_delta_W + j + 1
                n_total = n_tau_T * n_tau_delta_W

                # Progress callback every ~10 simulations (or at milestones)
                if progress_callback and (n_done % 10 == 0 or n_done == n_total):
                    percentage = int(10 + (n_done / n_total) * 80)  # Map to 10-90%
                    progress_callback(percentage)

                # Log progress every 50 simulations
                if n_done % 50 == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / n_done * (n_total - n_done)
                    logger.info(f"  Progress: {n_done}/{n_total} ({100*n_done/n_total:.1f}%) - "
                               f"ETA: {eta/60:.1f} min - {self.cache.get_stats()}")

        elapsed = time.time() - start_time
        logger.info(f"Grid search complete in {elapsed/60:.1f} min")
        logger.info(f"  {self.cache.get_stats()}")

        return tau_T_grid, tau_delta_W_grid, chi2_red_grid

    def find_minimum(self, tau_T_grid: np.ndarray, tau_delta_W_grid: np.ndarray,
                    chi2_red_grid: np.ndarray) -> Tuple[float, float, float]:
        """Find minimum chi2_red in grid. Returns (tau_T_best, tau_delta_W_best, chi2_red_best)"""
        i_min, j_min = np.unravel_index(np.argmin(chi2_red_grid), chi2_red_grid.shape)
        tau_T_best = tau_T_grid[i_min]
        tau_delta_W_best = tau_delta_W_grid[j_min]
        chi2_red_best = chi2_red_grid[i_min, j_min]

        logger.info(f"Minimum found: tau_T={tau_T_best:.3f} us, tau_Delta_W={tau_delta_W_best:.3f} us, "
                   f"chi2_red={chi2_red_best:.4f}")

        return tau_T_best, tau_delta_W_best, chi2_red_best


# Keep backward compatibility alias
HybridGridSearch = GridSearch
