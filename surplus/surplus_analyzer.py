"""
Surplus Analysis for Heterogeneous Systems

Simple 4-step algorithm:
1. Fit homogeneous model (Eq. 1) to late-time data (t > mask_time)
2. Subtract fit from raw data over ENTIRE time range → surplus
3. Fit homogeneous model to surplus signal
4. Fit heterogeneous bi-exponential (Eq. 4) to raw data using previous parameters

Author: SOLIS Development Team
Date: 2025-10-29
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import curve_fit

from utils.logger_config import get_logger
from core.kinetics_analyzer import KineticsAnalyzer

logger = get_logger(__name__)


@dataclass
class SurplusResult:
    """Container for surplus analysis results."""
    # Input
    time_us: np.ndarray
    intensity_raw: np.ndarray
    mask_time: float
    tau_T_guess: float

    # Step 1: Late-time fit (X)
    late_fit_A: float
    late_fit_tau_delta: float
    late_fit_tau_T: float
    late_fit_t0: float
    late_fit_y0: float
    late_fit_curve: np.ndarray
    late_fit_r2: float

    # Step 2: Surplus signal
    surplus_signal: np.ndarray

    # Step 3: Surplus fit (Y)
    surplus_fit_A: float
    surplus_fit_tau_delta: float
    surplus_fit_tau_T: float
    surplus_fit_t0: float
    surplus_fit_y0: float
    surplus_fit_curve: np.ndarray
    surplus_fit_r2: float

    # Step 4: Final heterogeneous fit
    final_alpha: float
    final_beta: float
    final_tau_delta_1: float
    final_tau_delta_2: float
    final_tau_T: float
    final_t0: float
    final_y0: float
    final_curve: np.ndarray
    final_r2: float

    @property
    def final_params(self) -> dict:
        """Return final parameters as dict for compatibility."""
        return {
            'alpha': self.final_alpha,
            'beta': self.final_beta,
            'tau_delta_1': self.final_tau_delta_1,
            'tau_delta_2': self.final_tau_delta_2,
            'tau_T': self.final_tau_T,
            't0': self.final_t0,
            'y0': self.final_y0
        }

    @property
    def final_r_squared(self) -> float:
        """Alias for compatibility with GUI/plotter."""
        return self.final_r2


def homogeneous_model(t, A, tau_delta, tau_T, t0, y0):
    """Homogeneous bi-exponential model (Eq. 1) with time offset."""
    t_shifted = t - t0
    t_shifted = np.where(t_shifted < 0, 0, t_shifted)

    if abs(tau_delta - tau_T) < 0.001:
        return np.full_like(t, y0)

    prefactor = A * tau_delta / (tau_delta - tau_T)
    result = prefactor * (np.exp(-t_shifted / tau_delta) - np.exp(-t_shifted / tau_T)) + y0
    return result


def homogeneous_model_no_t0(t, A, tau_delta, tau_T, y0):
    """Homogeneous bi-exponential model (Eq. 1) WITHOUT time offset (t0 = 0)."""
    if abs(tau_delta - tau_T) < 0.001:
        return np.full_like(t, y0)

    prefactor = A * tau_delta / (tau_delta - tau_T)
    result = prefactor * (np.exp(-t / tau_delta) - np.exp(-t / tau_T)) + y0
    return result


def heterogeneous_model(t, alpha, beta, tau_delta_1, tau_delta_2, tau_T, t0, y0):
    """Heterogeneous bi-exponential model (Eq. 4) with time offset."""
    t_shifted = t - t0
    t_shifted = np.where(t_shifted < 0, 0, t_shifted)

    # Component 1
    if abs(tau_delta_1 - tau_T) < 0.001:
        comp1 = 0
    else:
        prefactor_1 = alpha * tau_delta_1 / (tau_delta_1 - tau_T)
        comp1 = prefactor_1 * (np.exp(-t_shifted / tau_delta_1) - np.exp(-t_shifted / tau_T))

    # Component 2
    if abs(tau_delta_2 - tau_T) < 0.001:
        comp2 = 0
    else:
        prefactor_2 = beta * tau_delta_2 / (tau_delta_2 - tau_T)
        comp2 = prefactor_2 * (np.exp(-t_shifted / tau_delta_2) - np.exp(-t_shifted / tau_T))

    return comp1 + comp2 + y0


def heterogeneous_model_no_t0(t, alpha, beta, tau_delta_1, tau_delta_2, tau_T, y0):
    """Heterogeneous bi-exponential model (Eq. 4) WITHOUT time offset (t0 = 0)."""
    # Component 1
    if abs(tau_delta_1 - tau_T) < 0.001:
        comp1 = 0
    else:
        prefactor_1 = alpha * tau_delta_1 / (tau_delta_1 - tau_T)
        comp1 = prefactor_1 * (np.exp(-t / tau_delta_1) - np.exp(-t / tau_T))

    # Component 2
    if abs(tau_delta_2 - tau_T) < 0.001:
        comp2 = 0
    else:
        prefactor_2 = beta * tau_delta_2 / (tau_delta_2 - tau_T)
        comp2 = prefactor_2 * (np.exp(-t / tau_delta_2) - np.exp(-t / tau_T))

    return comp1 + comp2 + y0


def calculate_r2(y_true, y_pred):
    """Calculate R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def analyze_surplus(time: np.ndarray, intensity: np.ndarray,
                   mask_time: float, tau_T_guess: float,
                   tau_delta_fixed: float = 3.5) -> SurplusResult:
    """
    Perform surplus analysis for heterogeneous systems.

    IMPORTANT: For best results, run homogeneous kinetics analysis FIRST and use
    the resulting tau_T as tau_T_guess. This provides a good starting point.

    Parameters
    ----------
    time : np.ndarray
        Time array (μs)
    intensity : np.ndarray
        Raw intensity
    mask_time : float
        Time point for masking (fit from t > mask_time)
    tau_T_guess : float
        Initial guess for tau_T (should come from homogeneous analysis)
    tau_delta_fixed : float, optional
        Not actually fixed - used as initial guess only (default: 3.5)

    Returns
    -------
    SurplusResult
        Complete analysis results
    """
    logger.info("=" * 60)
    logger.info("SURPLUS ANALYSIS")
    logger.info(f"Mask time: {mask_time:.3f} μs")
    logger.info(f"τT guess: {tau_T_guess:.3f} μs")
    logger.info(f"τΔ fixed: {tau_delta_fixed:.3f} μs")
    logger.info("=" * 60)

    logger.info(f"Raw data: {len(time)} points, range [{np.nanmin(intensity):.1f}, {np.nanmax(intensity):.1f}]")
    logger.info(f"Time range: [{time[0]:.3f}, {time[-1]:.3f}] μs")
    mask_mask = time >= mask_time
    logger.info(f"Masked region: {np.sum(mask_mask)} points (t > {mask_time:.3f} μs)")

    # === STEP 1: Fit homogeneous model like kinetics analysis (mask sets interval) ===
    logger.info("\n[STEP 1] Fit homogeneous using data t > {:.3f} μs (model from t=0)".format(mask_time))

    # Create mask: fit only data where t > mask_time (like homogeneous kinetics)
    fitting_mask = time >= mask_time
    time_fit = time[fitting_mask]
    intensity_fit = intensity[fitting_mask]

    # Initial guesses
    A_guess = np.nanmax(intensity_fit) * 1.5
    tau_delta_guess = 3.5  # Use 3.5 as initial guess (typical for many systems)
    y0_guess = np.nanmean(intensity_fit[-20:])

    # Fit homogeneous model WITHOUT t0 (t0 = 0 fixed) - tau_delta is FREE
    p0 = [A_guess, tau_delta_guess, tau_T_guess, y0_guess]
    bounds = ([0, 0.1, 0.01, -np.inf], [np.inf, 50.0, 10.0, np.inf])

    popt, _ = curve_fit(homogeneous_model_no_t0, time_fit, intensity_fit,
                       p0=p0, bounds=bounds, maxfev=20000)
    A_late, tau_delta_late, tau_T_late, y0_late = popt
    t0_late = 0.0  # Fixed at zero

    # Model produces curve from t=0 to end
    late_fit_curve = homogeneous_model_no_t0(time, A_late, tau_delta_late, tau_T_late, y0_late)
    late_r2 = calculate_r2(intensity_fit,
                          homogeneous_model_no_t0(time_fit, A_late, tau_delta_late, tau_T_late, y0_late))

    logger.info(f"Late fit: A={A_late:.1f}, τΔ={tau_delta_late:.3f}, τT={tau_T_late:.3f}, t0=0.0 (fixed), R²={late_r2:.4f}")

    # === STEP 2: Calculate surplus over ENTIRE time range ===
    logger.info("\n[STEP 2] Calculate surplus = raw - late_fit")
    surplus = intensity - late_fit_curve
    logger.info(f"Surplus range: [{np.nanmin(surplus):.1f}, {np.nanmax(surplus):.1f}]")
    surplus_masked = surplus[fitting_mask]
    logger.info(f"Surplus in fit region (t > {mask_time:.3f}): [{np.nanmin(surplus_masked):.1f}, {np.nanmax(surplus_masked):.1f}]")
    logger.info(f"Surplus peak location: t = {time[np.argmax(surplus)]:.3f} μs, value = {np.nanmax(surplus):.1f}")

    # === STEP 3: Fit surplus with homogeneous model (ALL data, may have NaN) ===
    logger.info("\n[STEP 3] Fit homogeneous to surplus signal (full range)")

    # Fit surplus from t=0 to end (filter out NaN if present)
    valid_mask = np.isfinite(surplus)
    time_surplus = time[valid_mask]
    surplus_valid = surplus[valid_mask]

    logger.info(f"Surplus fitting: {len(surplus_valid)} valid points (filtered {np.sum(~valid_mask)} NaN/Inf)")

    # Initial guesses
    A_surplus_guess = np.nanmax(surplus_valid) * 1.5
    tau_delta_surplus_guess = 3.5  # Use 3.5 as initial guess (typical for many systems)
    y0_surplus_guess = np.nanmean(surplus_valid[-20:])

    # Fit homogeneous model WITHOUT t0 (t0 = 0 fixed)
    p0 = [A_surplus_guess, tau_delta_surplus_guess, tau_T_guess, y0_surplus_guess]
    bounds = ([0, 0.1, 0.01, -np.inf], [np.inf, 50.0, 10.0, np.inf])

    popt, _ = curve_fit(homogeneous_model_no_t0, time_surplus, surplus_valid,
                       p0=p0, bounds=bounds, maxfev=20000)
    A_surplus, tau_delta_surplus, tau_T_surplus, y0_surplus = popt
    t0_surplus = 0.0  # Fixed at zero

    # Model produces curve from t=0 to end
    surplus_fit_curve = homogeneous_model_no_t0(time, A_surplus, tau_delta_surplus, tau_T_surplus, y0_surplus)
    surplus_r2 = calculate_r2(surplus_valid,
                             homogeneous_model_no_t0(time_surplus, A_surplus, tau_delta_surplus, tau_T_surplus, y0_surplus))

    logger.info(f"Surplus fit: A={A_surplus:.1f}, τΔ={tau_delta_surplus:.3f}, τT={tau_T_surplus:.3f}, t0=0.0 (fixed), R²={surplus_r2:.4f}")

    # === STEP 4: Fit heterogeneous model (Eq. 4) to FULL raw data ===
    logger.info("\n[STEP 4] Fit heterogeneous model (Eq. 4) to FULL raw data")

    # Use parameters from Steps 1 and 3 as initial guesses
    alpha_guess = A_surplus
    beta_guess = A_late
    tau_delta_1_guess = tau_delta_surplus  # From surplus fit (fast component)
    tau_delta_2_guess = tau_delta_late     # From late-time fit (slow component)
    # Use higher tau_T to avoid local minimum (triplet lifetime should be longer)
    tau_T_final_guess = max(tau_T_late, tau_T_surplus)
    y0_final_guess = y0_late

    logger.info(f"Step 4 initial guesses: α={alpha_guess:.1f}, β={beta_guess:.1f}, "
               f"τΔ,1={tau_delta_1_guess:.3f}, τΔ,2={tau_delta_2_guess:.3f}, "
               f"τT={tau_T_final_guess:.3f} (max of steps 1&3)")

    # Fit heterogeneous model WITHOUT t0 (t0 = 0 fixed)
    # Relaxed bounds to allow better convergence and avoid local minima
    p0 = [alpha_guess, beta_guess, tau_delta_1_guess, tau_delta_2_guess,
          tau_T_final_guess, y0_final_guess]
    bounds_lower = [0, 0, 0.05, 0.05, 0.01, -np.inf]  # Relaxed tau_delta lower bound from 0.1 to 0.05
    bounds_upper = [np.inf, np.inf, 100.0, 100.0, 20.0, np.inf]  # Relaxed tau_delta upper from 50 to 100, tau_T from 10 to 20

    # Filter NaN/inf from raw data before fitting
    valid_mask = np.isfinite(intensity)
    time_valid = time[valid_mask]
    intensity_valid = intensity[valid_mask]

    logger.info(f"Heterogeneous fitting: {len(intensity_valid)} valid points (filtered {np.sum(~valid_mask)} NaN/Inf)")

    # Fit on valid raw data (NaN filtered) - increased maxfev for complex 6-parameter fit
    # Try multiple optimization strategies to avoid local minima
    try:
        # Strategy 1: Use Levenberg-Marquardt with relaxed bounds
        popt, _ = curve_fit(heterogeneous_model_no_t0, time_valid, intensity_valid,
                           p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=50000)
        alpha, beta, tau_delta_1, tau_delta_2, tau_T_final, y0_final = popt
        t0_final = 0.0  # Fixed at zero

        # Calculate initial fit quality
        initial_fit = heterogeneous_model_no_t0(time_valid, *popt)
        initial_r2 = calculate_r2(intensity_valid, initial_fit)

        logger.info(f"Initial fit R²: {initial_r2:.4f}")

        # If R² is poor, try alternative initial guess (swap fast/slow components)
        if initial_r2 < 0.95:
            logger.info("Trying alternative initial guess (swapping fast/slow components)...")
            p0_alt = [beta_guess, alpha_guess, tau_delta_2_guess, tau_delta_1_guess,
                     tau_T_final_guess, y0_final_guess]
            try:
                popt_alt, _ = curve_fit(heterogeneous_model_no_t0, time_valid, intensity_valid,
                                       p0=p0_alt, bounds=(bounds_lower, bounds_upper), maxfev=50000)
                alt_fit = heterogeneous_model_no_t0(time_valid, *popt_alt)
                alt_r2 = calculate_r2(intensity_valid, alt_fit)

                if alt_r2 > initial_r2:
                    logger.info(f"Alternative fit better: R² {alt_r2:.4f} > {initial_r2:.4f}")
                    popt = popt_alt
                    alpha, beta, tau_delta_1, tau_delta_2, tau_T_final, y0_final = popt
                else:
                    logger.info(f"Original fit better: R² {initial_r2:.4f} >= {alt_r2:.4f}")
            except Exception as e:
                logger.warning(f"Alternative fit failed: {e}")

    except Exception as e:
        logger.error(f"Heterogeneous fitting failed: {e}")
        raise

    # Final curve from t=0 to end (using FULL time array to generate complete curve)
    final_curve = heterogeneous_model_no_t0(time, alpha, beta, tau_delta_1, tau_delta_2,
                                           tau_T_final, y0_final)

    # Calculate R² using only valid points (same as fitting)
    final_r2 = calculate_r2(intensity_valid,
                           heterogeneous_model_no_t0(time_valid, alpha, beta, tau_delta_1, tau_delta_2,
                                                    tau_T_final, y0_final))

    logger.info(f"Final fit: α={alpha:.1f}, β={beta:.1f}, τΔ,1={tau_delta_1:.3f}, "
               f"τΔ,2={tau_delta_2:.3f}, τT={tau_T_final:.3f}, t0=0.0 (fixed), R²={final_r2:.4f}")

    logger.info("=" * 60)
    logger.info("SURPLUS ANALYSIS COMPLETE")
    logger.info("=" * 60)

    return SurplusResult(
        time_us=time,
        intensity_raw=intensity,
        mask_time=mask_time,
        tau_T_guess=tau_T_guess,
        late_fit_A=A_late,
        late_fit_tau_delta=tau_delta_late,  # FITTED, not fixed
        late_fit_tau_T=tau_T_late,
        late_fit_t0=t0_late,
        late_fit_y0=y0_late,
        late_fit_curve=late_fit_curve,
        late_fit_r2=late_r2,
        surplus_signal=surplus,
        surplus_fit_A=A_surplus,
        surplus_fit_tau_delta=tau_delta_surplus,  # FITTED, not fixed
        surplus_fit_tau_T=tau_T_surplus,
        surplus_fit_t0=t0_surplus,
        surplus_fit_y0=y0_surplus,
        surplus_fit_curve=surplus_fit_curve,
        surplus_fit_r2=surplus_r2,
        final_alpha=alpha,
        final_beta=beta,
        final_tau_delta_1=tau_delta_1,
        final_tau_delta_2=tau_delta_2,
        final_tau_T=tau_T_final,
        final_t0=t0_final,
        final_y0=y0_final,
        final_curve=final_curve,
        final_r2=final_r2
    )
