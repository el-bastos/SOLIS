#!/usr/bin/env python3
"""
Clean Kinetics Analyzer with 3-Step Workflow
Step 1: Remove baseline (lag phase points before spike)
Step 2: Create spike mask (artifact region)
Step 3: Fit Main and Literature models with chi-square goodness of fit
Step 4: DEPRECATED (removed - not needed for QY calculation)

ENHANCED: Chi-Square goodness of fit for Poisson-distributed TCSPC data
REFACTORED: Returns structured dataclasses instead of dictionaries (v2.5 Phase 2)
"""

import numpy as np
from scipy.optimize import curve_fit
from core.masking_methods import MaskingMethods, r2_score
from utils.logger_config import get_logger
from core.kinetics_dataclasses import (
    FitParameters, FitQuality, LiteratureModelResult,
    WorkflowInfo, KineticsResult, SNRResult
)
from core.snr_analyzer import SNRAnalyzer

logger = get_logger(__name__)


class KineticsAnalyzer(MaskingMethods):
    """Clean analyzer with 4-step workflow - fixed for module compatibility."""
    
    # Model detection thresholds
    MIN_TAU_T = 0.05  # μs - below this, consider single exponential
    MIN_PARAM_SEPARATION = 0.1  # μs - minimum |τΔ - τT| for biexponential
    MIN_BIEXP_R2 = 0.85  # R² threshold for acceptable biexponential fit
    
    def __init__(self, tau_delta_default=3.5):
        super().__init__(tau_delta_default)
        self.snr_analyzer = SNRAnalyzer()  # Phase 3A: SNR integration
    
    # Model functions (Literature formulation) - NO CHANGES
    def literature_biexponential(self, t, A, tau_delta, tau_T, y0):
        """Literature biexponential function f(t): A * (τΔ/(τΔ-τT)) * (e^(-t/τΔ) - e^(-t/τT)) + y0"""
        if abs(tau_delta - tau_T) < 1e-10:
            tau_T = tau_delta * 0.999
        
        factor = tau_delta / (tau_delta - tau_T)
        exp_term = np.exp(-t / tau_delta) - np.exp(-t / tau_T)
        return A * factor * exp_term + y0
    
    def main_biexponential(self, t, A, tau_delta, tau_T, t0, y0):
        """
        Main biexponential function f(t-t₀): A * (τΔ/(τΔ-τT)) * (e^(-(t-t₀)/τΔ) - e^(-(t-t₀)/τT)) + y0

        OPTIMIZED: Uses vectorized operations with minimal conditionals.
        """
        # Handle tau convergence (avoid division by zero)
        if abs(tau_delta - tau_T) < 1e-10:
            tau_T = tau_delta * 0.999

        # OPTIMIZED: Pre-compute factor outside exponentials
        factor = tau_delta / (tau_delta - tau_T)

        # OPTIMIZED: Vectorized time shift with clipping (faster than np.maximum for large arrays)
        t_shifted = t - t0
        t_shifted = np.clip(t_shifted, 0, None)  # Clip to [0, inf)

        # OPTIMIZED: Pre-compute exponentials once (avoiding redundant computation)
        exp_delta = np.exp(-t_shifted / tau_delta)
        exp_T = np.exp(-t_shifted / tau_T)

        # Final computation
        return A * factor * (exp_delta - exp_T) + y0
    
    def single_exponential(self, t, A, tau_delta, y0):
        """Single exponential: A * e^(-t/τΔ) + y0"""
        return A * np.exp(-t / tau_delta) + y0
    
    def single_exponential_with_t0(self, t, A, tau_delta, t0, y0):
        """
        Single exponential with offset: A * e^(-(t-t₀)/τΔ) + y0

        OPTIMIZED: Uses np.clip for faster clipping vs np.maximum.
        """
        # OPTIMIZED: np.clip is faster than np.maximum for this use case
        t_shifted = np.clip(t - t0, 0, None)
        return A * np.exp(-t_shifted / tau_delta) + y0
    
    def calculate_chi_square(self, residuals, intensities, n_parameters):
        """
        Calculate chi-square and reduced chi-square for Poisson-distributed data.

        OPTIMIZED VERSION: Uses vectorized operations and caches weighted residuals
        computation to avoid redundant calculations.

        For TCSPC data with Poisson statistics:
        χ² = Σ [(observed - predicted)² / variance]
           = Σ [(observed - predicted)² / observed]  (variance = mean for Poisson)
           = Σ [weighted_residual²]

        Args:
            residuals: Unweighted residuals (observed - predicted)
            intensities: Photon counts (observed data)
            n_parameters: Number of fitted parameters

        Returns:
            dict with chi_square, reduced_chi_square, degrees_of_freedom
        """
        # OPTIMIZED: Direct vectorized calculation without intermediate function call
        # weighted_residuals = residuals / sqrt(max(intensities, 1))
        # This avoids the overhead of calling calculate_weighted_residuals

        # Avoid division by zero: use max(intensity, 1) as variance estimate
        variances = np.maximum(intensities, 1.0)

        # Vectorized weighted residuals: residual / sqrt(variance)
        # Chi-square is sum of (residual² / variance)
        chi_square = np.sum(residuals**2 / variances)

        # Degrees of freedom = number of data points - number of parameters
        degrees_of_freedom = len(residuals) - n_parameters

        # Reduced chi-square (should be ~1.0 for good Poisson fits)
        reduced_chi_square = chi_square / degrees_of_freedom if degrees_of_freedom > 0 else np.nan

        return {
            'chi_square': chi_square,
            'reduced_chi_square': reduced_chi_square,
            'degrees_of_freedom': degrees_of_freedom
        }
    
    def detect_model_type(self, fit_result):
        """Detect if single exponential model should be used instead."""
        if not isinstance(fit_result, dict) or 'parameters' not in fit_result:
            return False, ["Invalid fit result structure"]
        
        params = fit_result['parameters']
        r2 = fit_result.get('fit_quality', {}).get('r2', 0)
        
        conditions = []
        
        if params.get('tau_T', 1) < self.MIN_TAU_T:
            conditions.append(f"τT ({params['tau_T']:.3f}) < {self.MIN_TAU_T}")
        
        if abs(params.get('tau_delta', 1) - params.get('tau_T', 1)) < self.MIN_PARAM_SEPARATION:
            conditions.append(f"|τΔ - τT| ({abs(params['tau_delta'] - params['tau_T']):.3f}) < {self.MIN_PARAM_SEPARATION}")
        
        if r2 < self.MIN_BIEXP_R2:
            conditions.append(f"R² ({r2:.3f}) < {self.MIN_BIEXP_R2}")
        
        return len(conditions) > 0, conditions
    
    def step1_remove_baseline(self, time, intensity, dataset_type='auto'):
        """
        Step 1: Process data according to dataset type classification.

        Args:
            time: Time array (μs)
            intensity: Intensity array
            dataset_type: One of 'lag_spike', 'spike_only', 'clean_signal', 'preprocessed', 'auto'

        Returns:
            Dictionary with time_experiment, intensity_experiment, and masking info
        """
        # Check for NaN (auto-detect preprocessed data)
        nan_mask = np.isnan(intensity)
        has_nan = np.any(nan_mask)

        if has_nan and dataset_type == 'auto':
            # Auto-detect: NaN present = preprocessed
            dataset_type = 'preprocessed'
            logger.info("Auto-detected dataset type: preprocessed (NaN present)")

        # CASE (d): Preprocessed - user already removed lag/spike
        if dataset_type == 'preprocessed':
            valid_indices = np.where(~nan_mask)[0]
            if len(valid_indices) == 0:
                raise ValueError("All intensity values are NaN")

            first_valid_idx = valid_indices[0]

            # NO SHIFTING - preserve user's timeline exactly!
            time_experiment = time.copy()
            intensity_experiment = intensity.copy()
            spike_end_experiment = first_valid_idx - 1 if first_valid_idx > 0 else 0

            return {
                'time_experiment': time_experiment,
                'intensity_experiment': intensity_experiment,
                'spike_start_original': 0,
                'spike_end_experiment': spike_end_experiment,
                'spike_start_time_original': time[0],
                'points_removed': 0,
                'has_nan': True,
                'dataset_type': 'preprocessed',
                'shifted': False
            }

        # CASE (c): Clean signal - no artifacts, no processing needed
        if dataset_type == 'clean_signal':
            return {
                'time_experiment': time.copy(),
                'intensity_experiment': intensity.copy(),
                'spike_start_original': 0,
                'spike_end_experiment': 0,
                'spike_start_time_original': time[0],
                'points_removed': 0,
                'has_nan': False,
                'dataset_type': 'clean_signal',
                'shifted': False
            }

        # For spike detection cases, call spike detector with dataset_type
        try:
            spike_result = self.spike_detector.detect_spikes(time, intensity, dataset_type)
            spike_region = spike_result.get('spike_region')
            lag_end_idx = spike_result.get('lag_end_idx', 0)

        except Exception as e:
            logger.error(f"Spike detection failed: {e}")
            # Fallback to legacy detection
            spike_start_idx, spike_end_idx = self.find_spikes(intensity)
            if spike_start_idx is None:
                spike_start_idx, spike_end_idx = 0, 10
            lag_end_idx = spike_start_idx
            spike_region = {'start': spike_start_idx, 'end': spike_end_idx}

        # CASE (a): Lag + Spike - SHIFT timeline to remove lag
        if dataset_type == 'lag_spike':
            # Bounds check: If lag extends to end of array, clamp it
            if lag_end_idx >= len(time):
                logger.warning(f"Lag region extends to end of data (idx={lag_end_idx}/{len(time)}). Clamping to last valid point.")
                lag_end_idx = len(time) - 1

            # Check if we have enough data left
            remaining_points = len(time) - lag_end_idx
            if remaining_points < 10:
                logger.error(f"Insufficient data after lag removal: only {remaining_points} points remaining (need at least 10)")
                return None

            if spike_region is None:
                # No spike found, but lag detected
                time_experiment = time[lag_end_idx:] - time[lag_end_idx]
                intensity_experiment = intensity[lag_end_idx:]
                spike_end_experiment = 0
            else:
                spike_start_idx = lag_end_idx  # Start from where lag ends
                spike_end_idx = min(spike_region['end'], len(time) - 1)  # Bounds check

                # SHIFT: Remove lag, new t=0 at lag end
                time_experiment = time[spike_start_idx:] - time[spike_start_idx]
                intensity_experiment = intensity[spike_start_idx:]
                spike_end_experiment = spike_end_idx - spike_start_idx

            return {
                'time_experiment': time_experiment,
                'intensity_experiment': intensity_experiment,
                'spike_start_original': lag_end_idx,
                'spike_end_experiment': spike_end_experiment,
                'spike_start_time_original': time[lag_end_idx],
                'points_removed': lag_end_idx,
                'has_nan': False,
                'dataset_type': 'lag_spike',
                'shifted': True
            }

        # CASE (b): Spike only - NO SHIFT, just mask spike
        elif dataset_type == 'spike_only':
            if spike_region is None:
                spike_end_experiment = 0
            else:
                spike_end_experiment = min(spike_region['end'], len(time) - 1)  # Bounds check

            # NO SHIFTING - keep original timeline
            return {
                'time_experiment': time.copy(),
                'intensity_experiment': intensity.copy(),
                'spike_start_original': 0,
                'spike_end_experiment': spike_end_experiment,
                'spike_start_time_original': time[0],
                'points_removed': 0,
                'has_nan': False,
                'dataset_type': 'spike_only',
                'shifted': False
            }

        # CASE 'auto': Legacy automatic mode
        else:
            if spike_region is None:
                spike_start_idx = spike_result.get('signal_start_idx', 0)
                spike_end_idx = spike_start_idx + 10
            else:
                spike_start_idx = lag_end_idx if lag_end_idx > 0 else spike_region.get('start', 0)
                spike_end_idx = min(spike_region['end'], len(time) - 1)  # Bounds check

            # Bounds check: Clamp spike_start_idx if it extends beyond array
            if spike_start_idx >= len(time):
                logger.warning(f"Spike start index extends to end of data (idx={spike_start_idx}/{len(time)}). Clamping to last valid point.")
                spike_start_idx = len(time) - 1

            # Check if we have enough data left
            remaining_points = len(time) - spike_start_idx
            if remaining_points < 10:
                logger.error(f"Insufficient data after lag removal: only {remaining_points} points remaining (need at least 10)")
                return None

            # Legacy behavior: shift if lag detected
            if spike_start_idx > 0:
                time_experiment = time[spike_start_idx:] - time[spike_start_idx]
                intensity_experiment = intensity[spike_start_idx:]
                spike_end_experiment = spike_end_idx - spike_start_idx
                # Clamp spike_end_experiment to valid range for the truncated array
                spike_end_experiment = max(0, min(spike_end_experiment, len(time_experiment) - 1))
                shifted = True
            else:
                time_experiment = time.copy()
                intensity_experiment = intensity.copy()
                spike_end_experiment = spike_end_idx
                shifted = False

            return {
                'time_experiment': time_experiment,
                'intensity_experiment': intensity_experiment,
                'spike_start_original': spike_start_idx,
                'spike_end_experiment': spike_end_experiment,
                'spike_start_time_original': time[spike_start_idx],
                'points_removed': spike_start_idx if shifted else 0,
                'has_nan': False,
                'dataset_type': 'auto',
                'shifted': shifted
            }
    
    def step2_create_spike_mask(self, time_experiment, spike_end_experiment):
        """
        Step 2: Create mask to exclude spike region from fitting (no timeline shifting).
        """
        # Mask spike region: fit after spike_end_experiment
        fitting_mask = np.arange(len(time_experiment)) > spike_end_experiment
        
        return {
            'fitting_mask': fitting_mask,
            'spike_duration': time_experiment[spike_end_experiment] if spike_end_experiment < len(time_experiment) else 0,
            'fitting_points': np.sum(fitting_mask),
            'total_points': len(time_experiment)
        }
    
    def step3_fit_both_models(self, time_experiment, intensity_experiment, fitting_mask, tau_delta_fixed):
        """
        Step 3: Fit both Main and Literature models on same timeline with same mask.
        ENHANCED: Now includes chi-square goodness of fit metrics.
        Baseline (y0) estimated from tail of full experimental timeline (not just fitted region).
        """
        t_fit = time_experiment[fitting_mask]
        I_fit = intensity_experiment[fitting_mask]

        if len(t_fit) < 10:
            raise ValueError(f"Insufficient fitting points: {len(t_fit)}")

        # Initial guesses - use tail baseline from full dataset
        # Use last 100-200 points of full intensity_experiment for better baseline estimate
        tail_length = min(200, len(intensity_experiment) // 4)  # 25% of data or 200 points max
        y0_guess = np.nanmean(intensity_experiment[-tail_length:])  # Use nanmean in case of NaN
        A_guess = np.max(I_fit) - y0_guess
        
        results = {}
        
        # Fit Main model f(t-t₀)
        try:
            main_initial = [A_guess, tau_delta_fixed, tau_delta_fixed * 0.3, 0.1, y0_guess]
            main_bounds = ([0, 0, 0, 0.0, -np.inf], [np.inf, np.inf, np.inf, 2.0, np.inf])
            
            main_popt, main_pcov = curve_fit(
                self.main_biexponential, t_fit, I_fit,
                p0=main_initial, bounds=main_bounds, maxfev=2000
            )
            
            main_pred = self.main_biexponential(t_fit, *main_popt)
            main_residuals = I_fit - main_pred
            main_r2 = r2_score(I_fit, main_pred)
            
            # Calculate chi-square (5 parameters for biexponential main model)
            main_chi2 = self.calculate_chi_square(main_residuals, I_fit, n_parameters=5)
            
            # Check if single exponential is better
            should_try_single, reasons = self.detect_model_type({
                'parameters': dict(zip(['A', 'tau_delta', 'tau_T', 't0', 'y0'], main_popt)),
                'fit_quality': {'r2': main_r2}
            })
            
            if should_try_single:
                # Try single exponential Main model
                try:
                    single_main_initial = [A_guess, tau_delta_fixed, 0.1, y0_guess]
                    single_main_bounds = ([0, 0, 0.0, -np.inf], [np.inf, np.inf, 2.0, np.inf])
                    
                    single_main_popt, _ = curve_fit(
                        self.single_exponential_with_t0, t_fit, I_fit,
                        p0=single_main_initial, bounds=single_main_bounds, maxfev=2000
                    )
                    
                    single_main_pred = self.single_exponential_with_t0(t_fit, *single_main_popt)
                    single_main_residuals = I_fit - single_main_pred
                    single_main_r2 = r2_score(I_fit, single_main_pred)
                    
                    # Calculate chi-square (4 parameters for single exponential main model)
                    single_main_chi2 = self.calculate_chi_square(single_main_residuals, I_fit, n_parameters=4)
                    
                    if single_main_r2 > main_r2:
                        # Use single exponential
                        main_popt = list(single_main_popt) + ['ND']  # [A, tau_delta, t0, y0, 'ND']
                        main_pred = single_main_pred
                        main_residuals = single_main_residuals
                        main_r2 = single_main_r2
                        main_chi2 = single_main_chi2
                        model_used = 'single_exponential'
                    else:
                        model_used = 'biexponential'
                except:
                    model_used = 'biexponential'
            else:
                model_used = 'biexponential'
            
            if model_used == 'single_exponential':
                results['main'] = {
                    'success': True,
                    'parameters': {
                        'A': main_popt[0],
                        'tau_delta': main_popt[1],
                        'tau_T': 'ND',
                        't0': main_popt[2],
                        'y0': main_popt[3]
                    },
                    'fit_quality': {
                        'r2': main_r2,
                        'residual_standard_error': np.sqrt(np.mean(main_residuals**2)),
                        'chi_square': main_chi2['chi_square'],
                        'reduced_chi_square': main_chi2['reduced_chi_square'],
                        'degrees_of_freedom': main_chi2['degrees_of_freedom'],
                        'n_fit_points': len(t_fit),
                        'model_used': 'single_exponential'
                    },
                    'fitting_data': {
                        't_fit': t_fit,
                        'I_fit': I_fit,
                        'I_pred': main_pred,
                        'residuals': main_residuals
                    }
                }
            else:
                results['main'] = {
                    'success': True,
                    'parameters': {
                        'A': main_popt[0],
                        'tau_delta': main_popt[1],
                        'tau_T': main_popt[2],
                        't0': main_popt[3],
                        'y0': main_popt[4]
                    },
                    'fit_quality': {
                        'r2': main_r2,
                        'residual_standard_error': np.sqrt(np.mean(main_residuals**2)),
                        'chi_square': main_chi2['chi_square'],
                        'reduced_chi_square': main_chi2['reduced_chi_square'],
                        'degrees_of_freedom': main_chi2['degrees_of_freedom'],
                        'n_fit_points': len(t_fit),
                        'model_used': 'biexponential'
                    },
                    'fitting_data': {
                        't_fit': t_fit,
                        'I_fit': I_fit,
                        'I_pred': main_pred,
                        'residuals': main_residuals
                    }
                }
            
        except Exception as e:
            results['main'] = {'success': False, 'error': str(e)}
        
        # Fit Literature model f(t)
        try:
            if results['main']['success'] and results['main']['parameters']['tau_T'] == 'ND':
                # Use single exponential Literature model
                lit_initial = [A_guess, tau_delta_fixed, y0_guess]
                lit_bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
                
                lit_popt, lit_pcov = curve_fit(
                    self.single_exponential, t_fit, I_fit,
                    p0=lit_initial, bounds=lit_bounds, maxfev=2000
                )
                
                lit_pred = self.single_exponential(t_fit, *lit_popt)
                lit_residuals = I_fit - lit_pred
                lit_r2 = r2_score(I_fit, lit_pred)
                
                # Calculate chi-square (3 parameters for single exponential literature model)
                lit_chi2 = self.calculate_chi_square(lit_residuals, I_fit, n_parameters=3)
                
                results['literature'] = {
                    'success': True,
                    'A': lit_popt[0],  # This is S₀
                    'tau_delta': lit_popt[1],
                    'tau_T': 'ND',
                    'y0': lit_popt[2],
                    'r_squared': lit_r2,
                    'residual_standard_error': np.sqrt(np.mean(lit_residuals**2)),
                    'chi_square': lit_chi2['chi_square'],
                    'reduced_chi_square': lit_chi2['reduced_chi_square'],
                    'degrees_of_freedom': lit_chi2['degrees_of_freedom'],
                    'n_fit_points': len(t_fit),
                    'model_used': 'single_exponential',
                    'fitting_data': {
                        't_fit': t_fit,
                        'I_fit': I_fit,
                        'I_pred': lit_pred,
                        'residuals': lit_residuals
                    }
                }
            else:
                # Use biexponential Literature model
                lit_initial = [A_guess, tau_delta_fixed, tau_delta_fixed * 0.3, y0_guess]
                lit_bounds = ([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
                
                lit_popt, lit_pcov = curve_fit(
                    self.literature_biexponential, t_fit, I_fit,
                    p0=lit_initial, bounds=lit_bounds, maxfev=2000
                )
                
                lit_pred = self.literature_biexponential(t_fit, *lit_popt)
                lit_residuals = I_fit - lit_pred
                lit_r2 = r2_score(I_fit, lit_pred)
                
                # Calculate chi-square (4 parameters for biexponential literature model)
                lit_chi2 = self.calculate_chi_square(lit_residuals, I_fit, n_parameters=4)
                
                results['literature'] = {
                    'success': True,
                    'A': lit_popt[0],  # This is S₀
                    'tau_delta': lit_popt[1],
                    'tau_T': lit_popt[2],
                    'y0': lit_popt[3],
                    'r_squared': lit_r2,
                    'residual_standard_error': np.sqrt(np.mean(lit_residuals**2)),
                    'chi_square': lit_chi2['chi_square'],
                    'reduced_chi_square': lit_chi2['reduced_chi_square'],
                    'degrees_of_freedom': lit_chi2['degrees_of_freedom'],
                    'n_fit_points': len(t_fit),
                    'model_used': 'biexponential',
                    'fitting_data': {
                        't_fit': t_fit,
                        'I_fit': I_fit,
                        'I_pred': lit_pred,
                        'residuals': lit_residuals
                    }
                }
                
        except Exception as e:
            results['literature'] = {'success': False, 'error': str(e)}
        
        return results

    def generate_full_curves(self, time_experiment, main_result, literature_result):
        """
        Generate full model curves for export and plotting.
        Step 4 removed - not needed for QY calculation.
        """
        curves = {}

        # Main model curve on experimental timeline
        if main_result['success']:
            params = main_result['parameters']
            if params['tau_T'] == 'ND':
                curves['main'] = self.single_exponential_with_t0(
                    time_experiment, params['A'], params['tau_delta'], params['t0'], params['y0']
                )
            else:
                curves['main'] = self.main_biexponential(
                    time_experiment, params['A'], params['tau_delta'], params['tau_T'], params['t0'], params['y0']
                )

        # Literature model curve on experimental timeline
        if literature_result['success']:
            if literature_result['tau_T'] == 'ND':
                curves['literature'] = self.single_exponential(
                    time_experiment, literature_result['A'], literature_result['tau_delta'], literature_result['y0']
                )
            else:
                curves['literature'] = self.literature_biexponential(
                    time_experiment, literature_result['A'], literature_result['tau_delta'],
                    literature_result['tau_T'], literature_result['y0']
                )

        return curves
    
    def calculate_auc(self, A, tau_delta, tau_T):
        """Calculate area under curve for biexponential model."""
        if tau_T == 'ND':
            return 'ND'
        try:
            factor = tau_delta / (tau_delta - tau_T)
            auc = A * factor * (tau_delta - tau_T)
            return auc
        except:
            return 'ND'
    
    def calculate_weighted_residuals(self, residuals, intensities):
        """
        Calculate weighted residuals for Poisson-distributed data (TCSPC).
        
        For photon counting, variance = mean (Poisson statistics).
        Weight = 1/σ = 1/√intensity
        
        Args:
            residuals: Observed - predicted values
            intensities: Photon counts (observed data)
        
        Returns:
            Weighted residuals for χ² calculation
        """
        if len(residuals) != len(intensities):
            return residuals
        
        weights = np.sqrt(np.maximum(np.abs(intensities), 1.0))
        return residuals / weights

    def fit_kinetics(self, time, intensity, tau_delta_fixed=None, custom_mask_end_time=None, dataset_type='auto'):
        """
        Main fitting function using clean 3-step workflow (Step 4 deprecated).

        Returns exact variable names and direct parameter access for other modules.
        Includes chi-square goodness of fit metrics for Poisson-distributed TCSPC data.

        Parameters
        ----------
        time : np.ndarray
            Time array (μs)
        intensity : np.ndarray
            Intensity array (photon counts)
        tau_delta_fixed : float
            Fixed tau_delta value (μs)
        custom_mask_end_time : float, optional
            User-specified mask endpoint time (μs). If provided, overrides automatic spike detection.
        dataset_type : str, optional
            Type of dataset - controls timeline shifting behavior:
            - 'lag_spike': Lag + Spike + Signal (will shift timeline to remove lag)
            - 'spike_only': Spike + Signal (no timeline shift, mask spike only)
            - 'clean_signal': Clean signal (no shift, no mask)
            - 'preprocessed': User removed artifacts (no shift, NaN masking only)
            - 'auto': Legacy automatic detection (default for backward compatibility)
        """
        if tau_delta_fixed is None:
            raise ValueError("tau_delta_fixed is required")

        # Store original data
        original_time = time.copy()
        original_intensity = intensity.copy()

        logger.info("Starting clean 3-step kinetics analysis (Step 4 deprecated)")
        if custom_mask_end_time is not None:
            logger.info(f"Custom mask correction applied: mask_end_time = {custom_mask_end_time:.4f} μs")

        # Phase 3A: SNR analysis (before workflow)
        snr_result = self.snr_analyzer.analyze_snr(time, intensity)
        logger.info(f"SNR Analysis: {snr_result.snr_db:.1f} dB ({snr_result.quality})")
        logger.info(f"Dataset type: {dataset_type}")

        # Step 1: Remove baseline (before spike_start) with dataset type awareness
        step1_result = self.step1_remove_baseline(time, intensity, dataset_type)

        # Check if step1 failed (insufficient data)
        if step1_result is None:
            logger.error("Step 1 failed: insufficient data after baseline removal")
            return None

        time_experiment = step1_result['time_experiment']
        intensity_experiment = step1_result['intensity_experiment']

        logger.info(f"Step 1: Removed {step1_result['points_removed']} baseline points, experiment starts at t=0")

        # Step 2: Create spike mask (no timeline shifting)
        # If custom_mask_end_time provided, find corresponding index
        if custom_mask_end_time is not None:
            # IMPORTANT: custom_mask_end_time is in ORIGINAL time coordinates
            # We need to find the index in original_time, then adjust to time_experiment
            spike_end_idx_original = np.searchsorted(original_time, custom_mask_end_time)
            spike_end_idx_original = min(spike_end_idx_original, len(original_time) - 1)

            # Adjust to time_experiment coordinates (account for removed baseline points)
            points_removed = step1_result['points_removed']
            spike_end_idx = spike_end_idx_original - points_removed

            # Ensure valid index in time_experiment
            spike_end_idx = max(0, min(spike_end_idx, len(time_experiment) - 1))

            logger.info(f"Using custom mask endpoint: {custom_mask_end_time:.4f} μs (original), "
                       f"index {spike_end_idx} in time_experiment ({time_experiment[spike_end_idx]:.4f} μs shifted)")
        else:
            spike_end_idx = step1_result['spike_end_experiment']

        step2_result = self.step2_create_spike_mask(time_experiment, spike_end_idx)
        fitting_mask = step2_result['fitting_mask']

        logger.info(f"Step 2: Masking spike for {step2_result['spike_duration']:.3f} us, fitting {step2_result['fitting_points']} points")

        # Step 3: Fit both models on same timeline
        step3_results = self.step3_fit_both_models(time_experiment, intensity_experiment, fitting_mask, tau_delta_fixed)
        main_result = step3_results['main']
        literature_result = step3_results['literature']

        if main_result['success']:
            main_params = main_result['parameters']
            main_quality = main_result['fit_quality']
            logger.info(f"Step 3 Main: A={main_params['A']:.1f}, tau_d={main_params['tau_delta']:.3f}, tau_T={main_params['tau_T']}, t0={main_params['t0']:.4f}, R2={main_quality['r2']:.3f}, chi2_red={main_quality['reduced_chi_square']:.3f}")

        if literature_result['success']:
            logger.info(f"Step 3 Literature: S0={literature_result['A']:.1f}, tau_d={literature_result['tau_delta']:.3f}, tau_T={literature_result['tau_T']}, R2={literature_result['r_squared']:.3f}, chi2_red={literature_result['reduced_chi_square']:.3f}")

        # Generate full curves
        curves = self.generate_full_curves(time_experiment, main_result, literature_result)
        
        # Calculate weighted residuals for main model
        main_weighted_residuals = None
        if 'main' in curves:
            main_residuals_temp = intensity_experiment - curves['main']
            main_weighted_residuals = self.calculate_weighted_residuals(main_residuals_temp, intensity_experiment)

        # Calculate weighted residuals for literature model
        literature_weighted_residuals = None
        if 'literature' in curves:
            literature_residuals_temp = intensity_experiment - curves['literature']
            literature_weighted_residuals = self.calculate_weighted_residuals(literature_residuals_temp, intensity_experiment)

        # Build dataclass result
        if main_result['success']:
            # Create FitParameters
            params = FitParameters(
                A=main_result['parameters']['A'],
                tau_delta=main_result['parameters']['tau_delta'],
                tau_T=main_result['parameters']['tau_T'],
                t0=main_result['parameters']['t0'],
                y0=main_result['parameters']['y0']
            )

            # Create FitQuality
            quality = FitQuality(
                r_squared=main_result['fit_quality']['r2'],
                chi_square=main_result['fit_quality']['chi_square'],
                reduced_chi_square=main_result['fit_quality']['reduced_chi_square'],
                residual_standard_error=main_result['fit_quality']['residual_standard_error'],
                n_fit_points=main_result['fit_quality']['n_fit_points'],
                degrees_of_freedom=main_result['fit_quality']['degrees_of_freedom'],
                model_used=main_result['fit_quality']['model_used']
            )

            # Create LiteratureModelResult
            lit_result = LiteratureModelResult(
                success=literature_result['success'],
                A=literature_result.get('A'),
                tau_delta=literature_result.get('tau_delta'),
                tau_T=literature_result.get('tau_T'),
                y0=literature_result.get('y0'),
                r_squared=literature_result.get('r_squared'),
                chi_square=literature_result.get('chi_square'),
                reduced_chi_square=literature_result.get('reduced_chi_square'),
                model_used=literature_result.get('model_used'),
                curve=curves.get('literature'),
                weighted_residuals=literature_weighted_residuals
            )

            # Create WorkflowInfo
            workflow = WorkflowInfo(
                method='clean_3_step_workflow',
                baseline_points_removed=step1_result['points_removed'],
                spike_duration=step2_result['spike_duration'],
                fitting_points=step2_result['fitting_points'],
                step4_status='deprecated'
            )

            # Create KineticsResult
            result = KineticsResult(
                parameters=params,
                fit_quality=quality,
                time_experiment_us=time_experiment,
                intensity_raw=intensity_experiment,
                main_curve=curves.get('main'),
                main_weighted_residuals=main_weighted_residuals,
                literature=lit_result,
                fitting_mask=fitting_mask,
                spike_region=~fitting_mask,
                workflow=workflow,
                raw_time=original_time,
                raw_intensity=original_intensity,
                snr_result=snr_result  # Phase 3A: Include SNR analysis
            )

            return result
        else:
            raise ValueError(f"Main fitting failed: {main_result['error']}")
    
    def robust_replicate_analysis(self, replicates_data, outlier_method='combined'):
        """Simplified replicate analysis - returns all replicates as valid."""
        n_total = len(replicates_data)
        valid_indices = list(range(n_total))
        
        outlier_summary = {
            "method": "none",
            "total_replicates": n_total,
            "outliers_removed": 0,
            "replicates_kept": n_total,
            "outlier_indices": [],
            "success_rate": 1.0
        }
        
        return valid_indices, outlier_summary