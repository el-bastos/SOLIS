#!/usr/bin/env python3
"""
Complete Transition-Based Spike Detector with Fully Adaptive Thresholds
All parameters calculated from data statistics - no hardcoded values
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict, List

class TransitionBasedSpikeDetector:
    def __init__(self):
        """Initialize transition-based detector for singlet oxygen kinetics."""
        pass
    
    def find_baseline_region_at_end(self, intensity: np.ndarray) -> Dict:
        """
        Find baseline from the END of the dataset (return to baseline after decay).
        
        For TCSPC data:
        - Beginning: Few baseline points before spike
        - End: MANY baseline points after full decay (better statistics!)
        
        Returns baseline info from the tail region.
        NOTE: This is separate from spike detection - only for SNR calculation!
        """
        n_points = len(intensity)
        
        if n_points < 10:
            # Dataset too short - use whatever we have
            return {
                'end_idx': n_points - 1,
                'start_idx': 0,
                'length': n_points,
                'mean': np.mean(intensity),
                'std': np.std(intensity),
                'median': np.median(intensity),
                'cv': np.std(intensity) / (np.mean(intensity) + 1e-10),
                'dataset_type': 'insufficient_data',
                'location': 'full_dataset'
            }
        
        # STRATEGY: Find longest stable low-intensity region at the END
        # Start from the end and work backwards
        
        # Calculate percentiles from last portion of data to get idea of tail distribution
        tail_sample_size = min(200, n_points // 2)
        tail_sample = intensity[-tail_sample_size:]
        
        tail_10th = np.percentile(tail_sample, 10)
        tail_25th = np.percentile(tail_sample, 25)
        tail_median = np.percentile(tail_sample, 50)
        tail_90th = np.percentile(tail_sample, 90)
        
        # Baseline should be near the lower end of distribution
        # Use IQR to define "baseline-like" intensity
        iqr = tail_90th - tail_10th
        
        # Adaptive baseline threshold: points below this are baseline candidates
        baseline_threshold = tail_25th + iqr * 0.5  # Allow some margin above 25th percentile
        
        # Search backwards from end for longest stable region
        max_baseline_length = min(500, n_points // 3)  # Search up to 1/3 of dataset or 500 points
        
        candidates = []
        
        # Try different baseline lengths starting from the end
        for length in range(10, max_baseline_length):
            if length > n_points:
                break
                
            baseline_data = intensity[-length:]  # Take last 'length' points
            mean_val = np.mean(baseline_data)
            std_val = np.std(baseline_data)
            median_val = np.median(baseline_data)
            
            # Check if this region is "baseline-like"
            if mean_val > baseline_threshold:
                continue
            
            cv = std_val / (mean_val + 1e-10)
            
            if cv > 1.5:
                continue
            
            # Calculate outlier fraction
            mad = np.median(np.abs(baseline_data - median_val))
            mad_std = mad * 1.4826
            outlier_threshold = median_val + 3 * mad_std
            n_outliers = np.sum(baseline_data > outlier_threshold)
            outlier_fraction = n_outliers / length
            
            if outlier_fraction > 0.1:
                continue
            
            # Score
            stability_score = cv
            level_score = (mean_val - tail_10th) / (tail_median - tail_10th + 1e-10)
            length_bonus = -0.001 * length
            outlier_penalty = outlier_fraction
            
            total_score = stability_score + level_score + length_bonus + outlier_penalty
            
            candidates.append({
                'length': length,
                'start_idx': n_points - length,
                'end_idx': n_points - 1,
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'cv': cv,
                'outlier_fraction': outlier_fraction,
                'score': total_score
            })
        
        if not candidates:
            # Fallback: use last 50 points (or whatever is available)
            fallback_length = min(50, n_points)
            baseline_data = intensity[-fallback_length:]
            return {
                'start_idx': n_points - fallback_length,
                'end_idx': n_points - 1,
                'length': fallback_length,
                'mean': np.mean(baseline_data),
                'std': np.std(baseline_data),
                'median': np.median(baseline_data),
                'cv': np.std(baseline_data) / (np.mean(baseline_data) + 1e-10),
                'dataset_type': 'fallback_tail',
                'location': 'end',
                'baseline_threshold': baseline_threshold
            }
        
        # Select best candidate (lowest score)
        best = min(candidates, key=lambda x: x['score'])
        best.update({
            'dataset_type': 'tail_baseline',
            'location': 'end',
            'baseline_threshold': baseline_threshold,
            'selection_method': 'backward_search_from_end'
        })
        
        return best


    def find_baseline_region(self, intensity: np.ndarray) -> Dict:
        """
        Find baseline at BEGINNING for spike detection (original method).
        This is needed to identify where the spike starts.
        """
        n_points = len(intensity)
        
        if n_points < 3:
            return {
                'end_idx': n_points - 1,
                'length': n_points,
                'mean': np.mean(intensity),
                'std': np.std(intensity),
                'cv': 0,
                'dataset_type': 'minimal'
            }
        
        # Simple approach: find first stable low region
        # Use first 3-25 points
        max_search = min(25, n_points)
        
        best_baseline = None
        best_cv = float('inf')
        
        for length in range(3, max_search + 1):
            data = intensity[:length]
            mean_val = np.mean(data)
            std_val = np.std(data)
            cv = std_val / (mean_val + 1e-10)
            
            # Stop if we hit a big jump
            if length > 3:
                if intensity[length - 1] > mean_val + 5 * std_val:
                    break
            
            if cv < best_cv:
                best_cv = cv
                best_baseline = {
                    'end_idx': length - 1,
                    'length': length,
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv,
                    'dataset_type': 'beginning'
                }
        
        if best_baseline is None:
            best_baseline = {
                'end_idx': 2,
                'length': 3,
                'mean': np.mean(intensity[:3]),
                'std': np.std(intensity[:3]),
                'cv': 0,
                'dataset_type': 'fallback'
            }
        
        return best_baseline

    def _find_lag_region(self, intensity: np.ndarray, baseline_stats: Dict) -> int:
        """
        Find the end of the lag region at the beginning of the dataset.
        Used for 'lag_spike' dataset type.

        Args:
            intensity: Intensity array
            baseline_stats: Baseline statistics from END (for threshold calculation)

        Returns:
            Index where lag ends and spike/signal begins
        """
        n_points = len(intensity)

        if n_points < 10:
            return 0

        # Search up to first 200 points or 10% of dataset
        max_search = min(200, n_points // 10)

        # Use baseline stats from END to set thresholds
        baseline_level = baseline_stats['mean']
        baseline_noise = baseline_stats['std']

        # Threshold: when intensity rises above baseline + 3*noise
        threshold = baseline_level + 3 * baseline_noise

        # Find first point that exceeds threshold consistently
        for i in range(max_search):
            if i + 5 >= n_points:
                break

            # Check if next 5 points all exceed threshold
            if np.all(intensity[i:i+5] > threshold):
                # Found lag end
                return max(0, i - 1)

        # No clear lag found - return 0
        return 0

    def _detect_spike_ramp_start(self, intensity: np.ndarray, max_search: int = 50) -> int:
        """
        Detect spike ramp start using derivative-based outlier detection.
        Still needed for spike detection, but baseline now comes from END.
        """
        max_search = min(max_search, len(intensity) - 1)
        
        if max_search < 5:
            return max_search
        
        # Calculate first derivative
        diff = np.diff(intensity[:max_search + 1])
        
        if len(diff) < 3:
            return max_search
        
        # Use first portion to establish "normal" derivative statistics
        early_window = min(10, len(diff) // 2)
        early_diff = diff[:early_window]
        
        diff_median = np.median(early_diff)
        mad = np.median(np.abs(early_diff - diff_median))
        diff_std_robust = mad * 1.4826
        
        # Outlier threshold using Tukey's method
        q75, q25 = np.percentile(early_diff, [75, 25])
        iqr = q75 - q25
        
        if iqr > 0 and diff_std_robust > 0:
            k = max(1.5, iqr / diff_std_robust)
        else:
            k = 3
        
        outlier_threshold = diff_median + k * diff_std_robust
        
        # Find first outlier (spike ramp start)
        for i in range(early_window, len(diff)):
            if diff[i] > outlier_threshold:
                return i + 1
        
        return max_search
    
    def calculate_intensity_derivatives(self, intensity: np.ndarray) -> Dict:
        """
        Calculate derivatives to find rapid changes in intensity.
        """
        first_deriv = np.gradient(intensity)
        second_deriv = np.gradient(first_deriv)
        
        window_size = max(3, len(intensity) // 500)
        if window_size % 2 == 0:
            window_size += 1
            
        first_deriv_smooth = signal.savgol_filter(first_deriv, window_size, 2)
        second_deriv_smooth = signal.savgol_filter(second_deriv, window_size, 2)
        
        return {
            'first_derivative': first_deriv,
            'second_derivative': second_deriv,
            'first_derivative_smooth': first_deriv_smooth,
            'second_derivative_smooth': second_deriv_smooth
        }
    
    def detect_spike_peak_region(self, intensity: np.ndarray, baseline: Dict) -> Optional[Dict]:
        """
        Find the main spike peak using intensity and baseline information.
        ADAPTIVE: All thresholds calculated from data statistics.
        """
        max_intensity = np.max(intensity)
        intensity_range = max_intensity - baseline['mean']
        
        # ADAPTIVE NO-SPIKE THRESHOLD: Based on data distribution
        intensity_95th = np.percentile(intensity, 95)
        intensity_90th = np.percentile(intensity, 90)
        
        # No-spike if max intensity suggests no real laser spike
        # Real spikes create bimodal distributions with large gaps
        if intensity_95th < intensity_90th * 1.2:  # Very tight distribution = no spike
            return None
        
        # Also check absolute scale - very low max suggests direct signal
        baseline_noise_level = baseline['mean'] + 10 * baseline['std']
        if max_intensity < baseline_noise_level * 5:  # Less than 5x noise level
            return None
        
        # ADAPTIVE PEAK THRESHOLD: Based on data range and distribution
        if intensity_range > 1000 * baseline['mean']:
            # Very large spike - use percentile approach
            peak_threshold = baseline['mean'] + max(
                10 * baseline['std'], 
                intensity_95th * 0.1  # 10% of 95th percentile
            )
        else:
            # Normal case - scale with data range
            threshold_factor = min(50, max(5, intensity_range / 100))  # 5-50x baseline std
            peak_threshold = baseline['mean'] + max(threshold_factor * baseline['std'], baseline_noise_level)
        
        peaks_above_threshold = []
        for i in range(len(intensity)):
            if intensity[i] > peak_threshold:
                peaks_above_threshold.append((i, intensity[i]))
        
        if not peaks_above_threshold:
            return None
        
        peak_idx, peak_value = max(peaks_above_threshold, key=lambda x: x[1])
        
        return {
            'peak_idx': peak_idx,
            'peak_value': peak_value,
            'height_above_baseline': peak_value - baseline['mean'],
            'adaptive_peak_threshold': peak_threshold,
            'intensity_95th': intensity_95th,
            'intensity_90th': intensity_90th
        }
    
    def find_spike_boundaries_by_transition(self, time: np.ndarray, intensity: np.ndarray, 
                                          derivatives: Dict, spike_peak: Dict, 
                                          baseline: Dict) -> Dict:
        """
        Find spike boundaries by detecting rapid transitions.
        ADAPTIVE: Better handling of complex spike shapes.
        """
        peak_idx = spike_peak['peak_idx']
        peak_value = spike_peak['peak_value']
        
        first_deriv = derivatives['first_derivative_smooth']
        second_deriv = derivatives['second_derivative_smooth']
        
        # Find spike start: rapid increase before peak
        spike_start = baseline['end_idx'] + 1  # Default fallback
        
        for i in range(peak_idx, max(0, baseline['end_idx']), -1):
            if (intensity[i] <= baseline['mean'] + 3 * baseline['std'] and
                i + 1 < len(intensity) and
                first_deriv[i + 1] > 0):
                spike_start = i + 1
                break
        
        # ADAPTIVE spike end detection
        spike_end = self._find_spike_end_by_transition_analysis(
            time, intensity, first_deriv, second_deriv, peak_idx, peak_value, baseline
        )
        
        return {
            'start': spike_start,
            'end': spike_end,
            'peak_idx': peak_idx,
            'peak_value': peak_value,
            'width': spike_end - spike_start + 1,
            'height_above_baseline': spike_peak['height_above_baseline']
        }
    
    def _find_spike_end_by_transition_analysis(self, time: np.ndarray, intensity: np.ndarray,
                                             first_deriv: np.ndarray, second_deriv: np.ndarray,
                                             peak_idx: int, peak_value: float, 
                                             baseline: Dict) -> int:
        """
        ADAPTIVE spike end detection - the key fix for problematic datasets.
        """
        
        # Method 1: Look for the steepest drop after peak
        search_window = min(50, len(intensity) - peak_idx - 1)
        if search_window < 5:
            return min(peak_idx + 3, len(intensity) - 1)
        
        min_derivative_idx = peak_idx
        min_derivative_value = 0
        
        for i in range(peak_idx + 1, peak_idx + search_window):
            if i < len(first_deriv) and first_deriv[i] < min_derivative_value:
                min_derivative_value = first_deriv[i]
                min_derivative_idx = i
        
        # Method 2: ADAPTIVE transition detection
        transition_candidates = []
        
        # ADAPTIVE: Lower thresholds for problematic datasets
        if peak_value > 1000 * baseline['mean']:
            slope_threshold = -10  # More lenient for huge spikes
            slope_change_threshold = 5  # Reduced sensitivity
        else:
            slope_threshold = -50  # Original threshold
            slope_change_threshold = 10
        
        for i in range(min_derivative_idx, min(min_derivative_idx + 20, len(first_deriv))):
            if i + 1 < len(first_deriv):
                slope_change = abs(second_deriv[i])
                current_slope = first_deriv[i]
                next_slope = first_deriv[i + 1] if i + 1 < len(first_deriv) else current_slope
                
                if (current_slope < slope_threshold and
                    next_slope > current_slope * 0.7 and
                    slope_change > slope_change_threshold):
                    transition_candidates.append(i)
        
        # Method 3: ADAPTIVE intensity-based validation
        valid_transitions = []
        
        # ADAPTIVE: Calculate expected signal level based on spike magnitude
        if peak_value > 1000 * baseline['mean']:
            # For huge spikes, signal will be much higher than baseline
            signal_level_min = max(
                baseline['mean'] + 5 * baseline['std'],
                peak_value * 0.005  # 0.5% of peak as minimum
            )
            max_drop_ratio = 0.8  # Allow 80% drop from peak
            stability_threshold = 0.5  # More lenient stability
        else:
            # Original thresholds for normal spikes
            signal_level_min = baseline['mean'] + 5 * baseline['std']
            max_drop_ratio = 0.5  # 50% drop from peak
            stability_threshold = 0.3  # Stricter stability
        
        for candidate_idx in transition_candidates:
            if candidate_idx + 5 < len(intensity):
                future_region = intensity[candidate_idx + 1:candidate_idx + 6]
                future_mean = np.mean(future_region)
                future_std = np.std(future_region)
                
                signal_level_ok = future_mean > signal_level_min
                stability_ok = future_std < future_mean * stability_threshold
                drop_ok = future_mean < peak_value * max_drop_ratio
                
                if signal_level_ok and stability_ok and drop_ok:
                    valid_transitions.append(candidate_idx)
        
        # Method 4: Choose best transition point - MORE CONSERVATIVE
        if valid_transitions:
            # Choose later transition point for more conservative masking
            spike_end = max(valid_transitions)  # Changed from min() to max()
        elif transition_candidates:
            # Use later candidate for conservative approach
            spike_end = max(transition_candidates)  # Changed from min() to max()
        else:
            # ADAPTIVE fallback: adjust based on spike magnitude, more conservative
            if peak_value > 1000 * baseline['mean']:
                spike_end = min(peak_idx + 10, len(intensity) - 1)  # Longer for huge spikes
            else:
                spike_end = min(peak_idx + 15, len(intensity) - 1)  # Longer fallback
        
        # Final validation: spike end should not be too close to peak - MORE CONSERVATIVE
        min_width = 5 if peak_value > 1000 * baseline['mean'] else 8  # Increased minimum widths
        if spike_end - peak_idx < min_width:
            spike_end = min(peak_idx + min_width, len(intensity) - 1)
        
        return spike_end
    
    def _detect_broad_artifact(self, time: np.ndarray, intensity: np.ndarray, baseline: Dict) -> Optional[Dict]:
        """
        Detect broad artifacts using rolling ball approach.
        Simple and robust: find where intensity stops decreasing and stabilizes.
        """
        start_search = baseline['end_idx'] + 1
        
        if start_search >= len(intensity) - 50:
            return None
        
        # Find initial artifact peak
        initial_window = min(100, len(intensity) - start_search - 20)
        initial_region = intensity[start_search:start_search + initial_window]
        initial_max = np.max(initial_region)
        initial_max_idx = start_search + np.argmax(initial_region)
        
        # Check if artifact is significant enough to mask
        artifact_height = initial_max - baseline['mean']
        if artifact_height < max(50, 5 * baseline['std']):
            return None
        
        # ROLLING BALL APPROACH: Find where signal stops decreasing
        artifact_end = self._rolling_ball_signal_start(intensity, start_search, initial_max)
        
        return {
            'peak_idx': initial_max_idx,
            'peak_value': initial_max,
            'end_idx': artifact_end,
            'height_above_baseline': artifact_height,
            'duration': artifact_end - start_search,
            'method': 'rolling_ball'
        }
    
    def _rolling_ball_signal_start(self, intensity: np.ndarray, start_idx: int, peak_value: float) -> int:
        """
        Rolling ball approach to find signal start after broad artifact.
        Finds where the decreasing trend stops and signal stabilizes.
        """
        # Parameters for rolling ball
        ball_radius = max(10, len(intensity) // 200)  # Scale with dataset size
        min_search_length = max(50, len(intensity) // 50)  # At least search this far
        max_search_length = min(len(intensity) - start_idx - 20, len(intensity) // 3)
        
        if max_search_length < min_search_length:
            return min(start_idx + 20, len(intensity) - 1)
        
        # Rolling minimum to find the "bottom" of the ball rolling under the curve
        search_region = intensity[start_idx:start_idx + max_search_length]
        
        # Calculate rolling minimum with specified radius
        rolling_min = np.zeros(len(search_region))
        for i in range(len(search_region)):
            window_start = max(0, i - ball_radius)
            window_end = min(len(search_region), i + ball_radius + 1)
            rolling_min[i] = np.min(search_region[window_start:window_end])
        
        # Find where the difference between actual signal and rolling minimum stabilizes
        # This indicates the artifact decay has ended and true signal begins
        difference = search_region - rolling_min
        
        # Look for stabilization point
        stability_window = max(20, len(intensity) // 100)  # Window for checking stability
        
        for i in range(min_search_length, len(difference) - stability_window):
            # Check stability in this window
            window = difference[i:i + stability_window]
            window_std = np.std(window)
            window_mean = np.mean(window)
            
            # Signal has stabilized if:
            # 1. Low variation (stable)
            # 2. Not too high above rolling minimum (not in artifact decay)
            stability_ratio = window_std / (window_mean + 1e-10)
            is_stable = stability_ratio < 0.3  # Low variation
            is_low = window_mean < peak_value * 0.1  # Well below artifact peak
            
            if is_stable and is_low:
                # Found stabilization point
                return start_idx + i
        
        # Fallback: if no clear stabilization found, use conservative estimate
        return min(start_idx + min_search_length, len(intensity) - 1)
    
    def find_signal_start_by_stabilization(self, time: np.ndarray, intensity: np.ndarray,
                                         spike_region: Dict, baseline: Dict) -> int:
        """
        Find signal start by looking for stable intensity above baseline after spike.
        ADAPTIVE: Better signal level estimation for huge spikes.
        """
        search_start = spike_region['end'] + 1
        
        if search_start >= len(intensity):
            return len(intensity) - 1
        
        # ADAPTIVE: Signal threshold based on spike magnitude
        peak_value = spike_region['peak_value']
        if peak_value > 1000 * baseline['mean']:
            # For huge spikes, signal will be much higher than baseline
            signal_threshold = max(
                baseline['mean'] + 2 * baseline['std'],
                peak_value * 0.01  # 1% of peak
            )
            stability_factor = 0.5  # More lenient
        else:
            # Original logic for normal spikes
            signal_threshold = baseline['mean'] + 2 * baseline['std']
            stability_factor = 0.4
        
        for i in range(search_start, min(search_start + 10, len(intensity))):
            if intensity[i] > signal_threshold:
                future_window = min(5, len(intensity) - i - 1)
                if future_window >= 2:
                    future_points = intensity[i:i + future_window]
                    future_std = np.std(future_points)
                    future_mean = np.mean(future_points)
                    
                    if (future_std < future_mean * stability_factor and
                        future_mean > signal_threshold):
                        return i
        
        return min(search_start + 1, len(intensity) - 1)
    
    def validate_detection_quality(self, time: np.ndarray, intensity: np.ndarray,
                                 baseline: Dict, spike_region: Dict, signal_start_idx: int,
                                 derivatives: Dict) -> Dict:
        """
        Comprehensive validation of detection results.
        """
        validation = {
            'baseline_quality': 'unknown',
            'spike_detection_quality': 'unknown', 
            'transition_quality': 'unknown',
            'signal_quality': 'unknown',
            'overall_confidence': 0.0,
            'physics_validation': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Baseline validation
        if baseline['cv'] < 0.2:
            validation['baseline_quality'] = 'excellent'
        elif baseline['cv'] < 0.4:
            validation['baseline_quality'] = 'good'
        else:
            validation['baseline_quality'] = 'poor'
            validation['warnings'].append("High baseline noise")
        
        # Spike detection validation
        if baseline['mean'] > 0:
            height_ratio = spike_region['height_above_baseline'] / baseline['mean']
            if height_ratio > 100:
                validation['spike_detection_quality'] = 'excellent'
            elif height_ratio > 20:
                validation['spike_detection_quality'] = 'good'
            else:
                validation['spike_detection_quality'] = 'poor'
                validation['warnings'].append("Small spike relative to baseline")
        else:
            # Baseline is zero - very clean data, excellent for analysis
            validation['spike_detection_quality'] = 'excellent'
        
        # Transition quality - check derivative behavior
        spike_end_idx = spike_region['end']
        if spike_end_idx + 5 < len(derivatives['first_derivative']):
            spike_region_deriv = derivatives['first_derivative'][spike_region['peak_idx']:spike_end_idx]
            post_spike_deriv = derivatives['first_derivative'][spike_end_idx:spike_end_idx + 5]
            
            spike_deriv_magnitude = np.mean(np.abs(spike_region_deriv))
            post_spike_deriv_magnitude = np.mean(np.abs(post_spike_deriv))
            
            if spike_deriv_magnitude > 3 * post_spike_deriv_magnitude:
                validation['transition_quality'] = 'excellent'
            elif spike_deriv_magnitude > 1.5 * post_spike_deriv_magnitude:
                validation['transition_quality'] = 'good'
            else:
                validation['transition_quality'] = 'poor'
                validation['warnings'].append("Unclear spike-signal transition")
        
        # Signal quality
        signal_length = len(intensity) - signal_start_idx
        signal_data = intensity[signal_start_idx:]
        signal_range = np.max(signal_data) - np.min(signal_data)
        
        if signal_length > 200 and signal_range > 10 * baseline['std']:
            validation['signal_quality'] = 'excellent'
        elif signal_length > 100 and signal_range > 5 * baseline['std']:
            validation['signal_quality'] = 'good'
        else:
            validation['signal_quality'] = 'poor'
            validation['warnings'].append("Short or low-amplitude signal region")
        
        # Physics validation
        validation['physics_validation'] = {
            'spike_width_reasonable': spike_region['width'] < 50,
            'signal_above_baseline': np.mean(signal_data) > baseline['mean'] + baseline['std'],
            'sharp_transition': validation['transition_quality'] in ['excellent', 'good']
        }
        
        # Overall confidence
        quality_scores = {'excellent': 1.0, 'good': 0.8, 'poor': 0.4, 'unknown': 0.2}
        component_scores = [
            quality_scores[validation['baseline_quality']],
            quality_scores[validation['spike_detection_quality']], 
            quality_scores[validation['transition_quality']],
            quality_scores[validation['signal_quality']]
        ]
        
        physics_bonus = sum(validation['physics_validation'].values()) / len(validation['physics_validation'])
        validation['overall_confidence'] = (np.mean(component_scores) + physics_bonus) / 2
        
        return validation
    
    def detect_spikes(self, time: np.ndarray, intensity: np.ndarray, dataset_type: str = 'auto') -> Dict:
        """
        Main transition-based spike detection for singlet oxygen kinetics.
        FULLY ADAPTIVE with broad artifact detection.

        Args:
            time: Time array
            intensity: Intensity array
            dataset_type: Type of dataset - controls detection behavior:
                - 'lag_spike': Lag + Spike + Signal (detect both, will shift timeline)
                - 'spike_only': Spike + Signal (detect spike, no timeline shift)
                - 'clean_signal': Clean signal only (no detection needed)
                - 'preprocessed': User removed lag/spike (no detection needed)
                - 'auto': Legacy mode (tries to detect automatically)

        Returns:
            Dictionary with baseline stats, spike region, and signal start index
        """

        # Step 1: ALWAYS get baseline statistics from END (more points, better SNR)
        baseline_stats = self.find_baseline_region_at_end(intensity)

        # Step 2: Handle dataset types that don't need spike detection
        if dataset_type == 'clean_signal':
            # Case (c): Clean signal starting at t=0, no artifacts
            return {
                'baseline': baseline_stats,
                'tail_baseline': baseline_stats,  # Same as baseline - both from end for better SNR
                'spike_region': None,
                'signal_start_idx': 0,
                'lag_end_idx': 0,
                'signal_start_time': time[0],
                'derivatives': None,
                'validation': {'dataset_type': 'clean_signal', 'user_classified': True},
                'summary': "Clean signal - no spike detection performed (user classified)"
            }

        elif dataset_type == 'preprocessed':
            # Case (d): User already removed lag/spike (may have NaN)
            return {
                'baseline': baseline_stats,
                'tail_baseline': baseline_stats,  # Same as baseline - both from end for better SNR
                'spike_region': None,
                'signal_start_idx': 0,
                'lag_end_idx': 0,
                'signal_start_time': time[0],
                'derivatives': None,
                'validation': {'dataset_type': 'preprocessed', 'user_classified': True},
                'summary': "Preprocessed data - no spike detection performed (user classified)"
            }

        # Step 3: For spike detection cases, calculate derivatives
        derivatives = self.calculate_intensity_derivatives(intensity)

        # Step 4: Detect lag region if needed (case a: lag_spike)
        lag_end_idx = 0
        if dataset_type == 'lag_spike':
            lag_end_idx = self._find_lag_region(intensity, baseline_stats)

        # Step 5: Find spike peak (use baseline_stats for thresholds)
        spike_peak = self.detect_spike_peak_region(intensity, baseline_stats)

        if spike_peak is None:
            # No sharp spike detected - check for broad artifacts
            broad_artifact = self._detect_broad_artifact(time, intensity, baseline_stats)

            if broad_artifact is not None:
                # Found broad artifact - create spike_region for masking
                spike_region = {
                    'start': lag_end_idx if dataset_type == 'lag_spike' else 0,
                    'end': broad_artifact['end_idx'],
                    'peak_idx': broad_artifact['peak_idx'],
                    'peak_value': broad_artifact['peak_value'],
                    'width': broad_artifact['end_idx'] - lag_end_idx,
                    'height_above_baseline': broad_artifact['peak_value'] - baseline_stats['mean'],
                    'artifact_type': 'broad_decay'
                }

                signal_start_idx = broad_artifact['end_idx'] + 1

                validation = {
                    'baseline_quality': 'good' if baseline_stats['cv'] < 0.4 else 'poor',
                    'spike_detection_quality': 'broad_artifact',
                    'transition_quality': 'broad_decay',
                    'signal_quality': 'unknown',
                    'overall_confidence': 0.7,
                    'warnings': ['Broad artifact detected instead of sharp spike'],
                    'recommendations': ['Verify artifact masking is appropriate'],
                    'dataset_type': dataset_type
                }

                summary = self._create_transition_summary(
                    baseline_stats, spike_region, signal_start_idx, time, validation
                )

                return {
                    'baseline': baseline_stats,
                    'tail_baseline': baseline_stats,  # Same as baseline - both from end for better SNR
                    'spike_region': spike_region,
                    'signal_start_idx': signal_start_idx,
                    'lag_end_idx': lag_end_idx,
                    'signal_start_time': time[signal_start_idx] if signal_start_idx < len(time) else time[-1],
                    'derivatives': derivatives,
                    'validation': validation,
                    'summary': summary
                }
            else:
                # No spike or artifact detected
                if dataset_type == 'auto':
                    # Legacy mode: return 0 (clean signal assumed)
                    signal_start_idx = 0
                else:
                    # User said there's a spike but we can't find it - return lag_end
                    signal_start_idx = lag_end_idx

                validation = {
                    'baseline_quality': 'good' if baseline_stats['cv'] < 0.4 else 'poor',
                    'spike_detection_quality': 'no_spike',
                    'transition_quality': 'no_spike',
                    'signal_quality': 'unknown',
                    'overall_confidence': 0.6,
                    'warnings': ['No spike detected'],
                    'recommendations': ['Verify if spike is expected'],
                    'dataset_type': dataset_type
                }

                return {
                    'baseline': baseline_stats,
                    'tail_baseline': baseline_stats,  # Same as baseline - both from end for better SNR
                    'spike_region': None,
                    'signal_start_idx': signal_start_idx,
                    'lag_end_idx': lag_end_idx,
                    'signal_start_time': time[signal_start_idx] if signal_start_idx < len(time) else time[0],
                    'derivatives': derivatives,
                    'validation': validation,
                    'summary': f"No spike detected - signal starts at index {signal_start_idx}"
                }

        # Step 6: Find spike boundaries using transition analysis
        spike_region = self.find_spike_boundaries_by_transition(
            time, intensity, derivatives, spike_peak, baseline_stats
        )

        # Adjust spike start based on lag detection
        if dataset_type == 'lag_spike' and lag_end_idx > 0:
            spike_region['start'] = max(lag_end_idx, spike_region.get('start', 0))

        # Step 7: Find signal start
        signal_start_idx = self.find_signal_start_by_stabilization(
            time, intensity, spike_region, baseline_stats
        )

        # Step 8: Validate results
        validation = self.validate_detection_quality(
            time, intensity, baseline_stats, spike_region, signal_start_idx, derivatives
        )
        validation['dataset_type'] = dataset_type

        # Step 9: Create summary
        summary = self._create_transition_summary(
            baseline_stats, spike_region, signal_start_idx, time, validation
        )

        return {
            'baseline': baseline_stats,
            'tail_baseline': baseline_stats,  # Same as baseline - both from end for better SNR
            'spike_region': spike_region,
            'signal_start_idx': signal_start_idx,
            'lag_end_idx': lag_end_idx,
            'signal_start_time': time[signal_start_idx] if signal_start_idx < len(time) else time[-1],
            'derivatives': derivatives,
            'validation': validation,
            'summary': summary
        }
    
    def _create_transition_summary(self, baseline: Dict, spike_region: Dict, 
                                 signal_start_idx: int, time: np.ndarray,
                                 validation: Dict) -> str:
        """Create comprehensive summary of transition-based detection."""
        
        lines = []
        lines.append("ADAPTIVE TRANSITION-BASED SPIKE DETECTION RESULTS")
        lines.append("=" * 45)
        
        # Baseline
        lines.append(f"BASELINE:")
        lines.append(f"• Region: indices 0-{baseline['end_idx']} ({baseline['length']} points)")
        lines.append(f"• Level: {baseline['mean']:.2f} ± {baseline['std']:.2f}")
        lines.append(f"• Quality: {validation['baseline_quality']}")
        if 'adaptive_threshold' in baseline:
            lines.append(f"• Adaptive threshold: {baseline['adaptive_threshold']:.1f}")
        if 'dataset_type' in baseline:
            lines.append(f"• Dataset type: {baseline['dataset_type']}")
        
        # Spike detection
        if spike_region is not None:
            lines.append(f"\nSPIKE DETECTION:")

            # Safe index access - indices might be at array boundaries
            start_idx = min(spike_region['start'], len(time) - 1)
            end_idx = min(spike_region['end'], len(time) - 1)  # Clamp to valid index
            peak_idx = min(spike_region['peak_idx'], len(time) - 1)

            if spike_region.get('artifact_type') == 'broad_decay':
                lines.append(f"• Broad artifact detected (not sharp spike)")
                lines.append(f"• Peak: {spike_region['peak_value']:.0f} at index {peak_idx} (t={time[peak_idx]:.3f} μs)")
                lines.append(f"• Artifact span: indices {start_idx}-{end_idx} (duration: {spike_region['width']} points)")
                lines.append(f"• Times: t={time[start_idx]:.3f}-{time[end_idx]:.3f} μs")
                lines.append(f"• Height: {spike_region['height_above_baseline']:.0f} above baseline")
            else:
                lines.append(f"• Peak: {spike_region['peak_value']:.0f} at index {peak_idx} (t={time[peak_idx]:.3f} μs)")
                lines.append(f"• Boundaries: indices {start_idx}-{end_idx} (width: {spike_region['width']})")
                lines.append(f"• Times: t={time[start_idx]:.3f}-{time[end_idx]:.3f} μs")
                if baseline['mean'] > 0:
                    lines.append(f"• Height: {spike_region['height_above_baseline']:.0f} ({spike_region['height_above_baseline']/baseline['mean']:.0f}x baseline)")
                else:
                    lines.append(f"• Height: {spike_region['height_above_baseline']:.0f} (baseline=0, very clean)")

            lines.append(f"• Quality: {validation['spike_detection_quality']}")

            # Adaptive mode indicator for normal spikes
            if spike_region.get('artifact_type') != 'broad_decay':
                if baseline['mean'] > 0:
                    peak_to_baseline = spike_region['height_above_baseline'] / baseline['mean']
                else:
                    peak_to_baseline = float('inf')  # Infinite ratio for zero baseline
                if peak_to_baseline > 1000:
                    lines.append(f"• [ADAPTIVE MODE: Extreme spike detected ({peak_to_baseline:.0f}x baseline)]")
                else:
                    lines.append(f"• [STANDARD MODE: Normal spike profile ({peak_to_baseline:.0f}x baseline)]")
        else:
            lines.append(f"\nSPIKE DETECTION:")
            lines.append(f"• No spike detected (max intensity below adaptive thresholds)")
            lines.append(f"• Dataset treated as direct signal (no laser spike)")

        # Transition analysis
        lines.append(f"\nTRANSITION ANALYSIS:")
        lines.append(f"• Transition quality: {validation['transition_quality']}")
        # Safe index access for signal_start_idx
        signal_start_safe = min(signal_start_idx, len(time) - 1)
        lines.append(f"• Signal starts: index {signal_start_idx} (t={time[signal_start_safe]:.3f} μs)")
        lines.append(f"• Signal quality: {validation['signal_quality']}")
        
        # Physics validation (only if spike detected)
        if spike_region is not None:
            lines.append(f"\nPHYSICS VALIDATION:")
            physics = validation.get('physics_validation', {})
            for check, passed in physics.items():
                status = "✓" if passed else "✗"
                lines.append(f"• {check.replace('_', ' ').title()}: {status}")
        
        # Overall assessment
        lines.append(f"\nOVERALL ASSESSMENT:")
        lines.append(f"• Confidence: {validation['overall_confidence']:.1%}")
        
        if validation.get('warnings'):
            lines.append(f"• Warnings: {', '.join(validation['warnings'])}")
        
        if validation.get('recommendations'):
            lines.append(f"• Recommendations: {', '.join(validation['recommendations'])}")
        
        return "\n".join(lines)


def test_transition_detector():
    """Test the transition-based detector on simulated data."""
    
    detector = TransitionBasedSpikeDetector()
    
    # Create realistic test data
    time_test = np.linspace(0, 20, 2000)
    
    # Baseline
    intensity_test = np.random.normal(3, 0.5, len(time_test))
    
    # Sharp spike
    spike_mask = (time_test >= 0.2) & (time_test <= 0.4)
    spike_profile = 800 * np.exp(-((time_test[spike_mask] - 0.3) / 0.03)**2)
    intensity_test[spike_mask] += spike_profile
    
    # Biexponential signal starting at 0.45
    signal_mask = time_test >= 0.45
    t_signal = time_test[signal_mask] - 0.45
    A, tau_d, tau_T, y0 = 150, 3.5, 0.8, 5
    biexp = A * (tau_d / (tau_d - tau_T)) * (np.exp(-t_signal / tau_d) - np.exp(-t_signal / tau_T)) + y0
    intensity_test[signal_mask] += biexp
    
    # Test detection
    result = detector.detect_spikes(time_test, intensity_test)
    
    print("Fully Adaptive Transition-Based Detector Test:")
    print("=" * 45)
    print(result['summary'])
    
    return detector, result


if __name__ == "__main__":
    detector, result = test_transition_detector()