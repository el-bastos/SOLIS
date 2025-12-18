#!/usr/bin/env python3
"""
Masking Methods Module - Simplified for Time Shift Approach
Maintains compatibility while using simpler spike_start->t=0 + mask spike region approach
"""

import numpy as np
from scipy.optimize import curve_fit
from core.core_fitting import CoreFittingMethods, r2_score
from utils.logger_config import get_logger

logger = get_logger(__name__)

class MaskingMethods(CoreFittingMethods):
    def __init__(self, tau_delta_default=3.5):
        super().__init__(tau_delta_default)
        
        # Import spike detector
        try:
            from core.spike_detector_last import TransitionBasedSpikeDetector
            self.spike_detector = TransitionBasedSpikeDetector()
        except ImportError:
            raise ImportError("spike_detector.py is required")
    
    def find_spikes(self, intensity, threshold_factor=0.8):
        """Legacy spike detection interface."""
        time_dummy = np.linspace(0, len(intensity) * 0.01, len(intensity))
        
        try:
            result = self.spike_detector.detect_spikes(time_dummy, intensity)
            spike_region = result.get('spike_region')
            
            if spike_region is None:
                return None, None
            
            return spike_region['start'], spike_region['end']
        except:
            max_idx = np.argmax(intensity)
            return max(0, max_idx - 5), min(len(intensity) - 1, max_idx + 10)
    
    def find_spikes_from_replicate_info(self, intensity, replicate_results):
        """Find spike region using information from replicate fits."""
        spike_starts = []
        spike_ends = []
        
        for result in replicate_results:
            if result.get('spike_start') is not None and result.get('spike_end') is not None:
                spike_starts.append(result['spike_start'])
                spike_ends.append(result['spike_end'])
        
        if not spike_starts:
            return self.find_spikes(intensity)
        
        median_spike_start = int(np.median(spike_starts))
        median_spike_end = int(np.median(spike_ends))
        
        if (median_spike_start >= 0 and median_spike_end < len(intensity) and 
            median_spike_start <= median_spike_end):
            return median_spike_start, median_spike_end
        else:
            return self.find_spikes(intensity)
    
    def fit_with_masking(self, time, intensity, extra_points=3, initial_guess=None, dataset_type='auto'):
        """
        Legacy interface that now uses time-shift approach internally.
        Handles NaN values at the beginning by extending the mask.

        Args:
            time: Time array
            intensity: Intensity array
            extra_points: Extra points to include in mask
            initial_guess: Initial parameter guess
            dataset_type: Dataset classification ('auto', 'lag_spike', 'spike_only', 'clean_signal', 'preprocessed')
        """
        if initial_guess is None:
            initial_guess = [1500, self.tau_delta_default, 1.0, 0.4, 5]

        try:
            # Check for NaN at the beginning of intensity array
            nan_mask = np.isnan(intensity)
            has_nan = np.any(nan_mask)

            if has_nan and dataset_type == 'auto':
                dataset_type = 'preprocessed'

            # CASE (d): Preprocessed - user removed artifacts
            if dataset_type == 'preprocessed':
                valid_indices = np.where(~nan_mask)[0]
                if len(valid_indices) == 0:
                    raise ValueError("All intensity values are NaN")
                nan_end_idx = valid_indices[0]

                # NO TIME SHIFTING - preserve user timeline
                fitting_mask = ~nan_mask
                time_shifted = time.copy()
                spike_start_idx = 0
                spike_end_idx = nan_end_idx

            # CASE (c): Clean signal - no masking needed
            elif dataset_type == 'clean_signal':
                fitting_mask = np.ones(len(time), dtype=bool)
                time_shifted = time.copy()
                spike_start_idx = 0
                spike_end_idx = 0

            else:
                # Detect spike region with dataset_type
                try:
                    spike_result = self.spike_detector.detect_spikes(time, intensity, dataset_type)
                    spike_region = spike_result.get('spike_region')
                    lag_end_idx = spike_result.get('lag_end_idx', 0)

                    if spike_region is None:
                        spike_start_idx = spike_result.get('signal_start_idx', 0)
                        spike_end_idx = spike_start_idx + 10
                    else:
                        spike_start_idx = lag_end_idx if dataset_type == 'lag_spike' else spike_region.get('start', 0)
                        spike_end_idx = spike_region['end']

                    # Bounds check for array indices
                    spike_start_idx = min(spike_start_idx, len(time) - 1)
                    spike_end_idx = min(spike_end_idx, len(time) - 1)

                except Exception:
                    spike_start_idx, spike_end_idx = self.find_spikes(intensity)
                    if spike_start_idx is None:
                        spike_start_idx, spike_end_idx = 0, 10

                # CASE (a): Lag + Spike - SHIFT timeline
                if dataset_type == 'lag_spike':
                    time_shifted = time - time[spike_start_idx]
                    spike_end_shifted = time_shifted[spike_end_idx]
                    fitting_mask = time_shifted > spike_end_shifted

                # CASE (b): Spike only - NO SHIFT, mask spike
                elif dataset_type == 'spike_only':
                    time_shifted = time.copy()
                    spike_end_shifted = time_shifted[spike_end_idx]
                    fitting_mask = time_shifted > spike_end_shifted

                # CASE 'auto': Legacy behavior
                else:
                    time_shifted = time - time[spike_start_idx]
                    spike_end_shifted = time_shifted[spike_end_idx]
                    fitting_mask = time_shifted > spike_end_shifted
            
            # Extract fitting data
            t_fit = time_shifted[fitting_mask]
            I_fit = intensity[fitting_mask]
            
            if len(t_fit) < 10:
                raise ValueError(f"Insufficient data points: {len(t_fit)}")
            
            # Fit with bounds - FIXED: t0 must be >= 0
            bounds = ([0, 0.1, 0.05, 0.0, -np.inf], [np.inf, 10, 8, 2.0, np.inf])
            #                     ^^^^ t0 >= 0.0 (was -1.0)
            
            # Ensure initial guess respects bounds
            initial_guess[3] = max(0.0, initial_guess[3])  # Force t0 >= 0
            
            popt, pcov = curve_fit(
                self.literature_biexponential_free,
                t_fit, I_fit,
                p0=initial_guess,
                bounds=bounds,
                maxfev=1000
            )
            
            I_pred = self.literature_biexponential_free(t_fit, *popt)
            r2 = r2_score(I_fit, I_pred)
            
            residuals = I_fit - I_pred
            mse = np.mean(residuals**2)
            residual_se = np.sqrt(mse)
            
            return {
                "parameters": {
                    "A": popt[0], "tau_delta": popt[1], "tau_T": popt[2],
                    "t0": popt[3], "y0": popt[4]
                },
                "fit_quality": {
                    "r2": r2, "residual_standard_error": residual_se, 
                    "n_fit_points": len(t_fit)
                },
                "fitting_info": {
                    "signal_start_time": time[spike_end_idx],
                    "signal_start_idx": spike_end_idx,
                    "masked_points": 0,  # Not used in time-shift approach
                    "method": "time_shift_compatibility"
                },
                "fitting_data": {
                    "t_fit": t_fit, "I_fit": I_fit, "I_pred": I_pred,
                    "residuals": residuals, "fitting_indices": np.where(fitting_mask)[0]
                }
            }
            
        except Exception as e:
            raise ValueError(f"Time-shift approach failed: {e}")

    def scan_masking_range(self, time, intensity, max_extra_points=3, initial_guess=None):
        """Updated scan_masking_range to use time-shift approach."""
        max_extra_points = min(max_extra_points, 3)
        
        try:
            # Use single fit approach for time-shift
            result = self.fit_with_masking(time, intensity, extra_points=0, initial_guess=initial_guess)
            
            return {
                "best_result": result,
                "best_masking": 0,
                "best_r2": result["fit_quality"]["r2"],
                "scan_results": [(0, result["fit_quality"]["r2"])]
            }
            
        except Exception as e:
            raise ValueError(f"Time-shift masking scan failed: {e}")

    def trim_data_to_pulse_start(self, time, intensity):
        """Trim data to start from actual pulse using time-shift approach."""
        try:
            # Detect spike region
            spike_result = self.spike_detector.detect_spikes(time, intensity)
            
            if spike_result is None or spike_result.get('spike_region') is None:
                pulse_start_idx = spike_result.get('signal_start_idx', 0) if spike_result else 0
                pulse_start_time = time[pulse_start_idx]
                spike_result = {
                    'spike_region': None,
                    'signal_start_time': pulse_start_time,
                    'signal_start_idx': pulse_start_idx,
                    'method': 'fallback_no_pulse'
                }
            else:
                pulse_start_idx = spike_result['spike_region']['start']
                pulse_start_time = time[pulse_start_idx]
        
        except Exception as e:
            logger.warning(f"Spike detection failed ({e}). Using simple fallback.")
            pulse_start_idx = 0
            pulse_start_time = time[0]
            spike_result = {
                'spike_region': None,
                'signal_start_time': time[0],
                'signal_start_idx': 0,
                'method': 'fallback_error'
            }
        
        # Time shift: pulse_start -> t=0
        time_shifted = time - pulse_start_time
        intensity_trimmed = intensity.copy()
        
        return {
            'time_trimmed': time_shifted,
            'intensity_trimmed': intensity_trimmed,
            'pulse_start_idx': pulse_start_idx,
            'pulse_start_time': pulse_start_time,
            'time_trimmed_original': time.copy(),
            'spike_result': spike_result
        }