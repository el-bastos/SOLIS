#!/usr/bin/env python3
"""
SNR Analyzer Module for SOLIS
==============================

Extracts signal-to-noise ratio analysis from SNR_and_exponential_fitting.py.
Uses spike detection to identify and exclude laser artifacts, then calculates
SNR using tail baseline statistics.

Phase 3A refactoring - modular architecture.
"""

import numpy as np
from core.spike_detector_last import TransitionBasedSpikeDetector
from core.kinetics_dataclasses import SNRResult
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SNRAnalyzer:
    """
    Analyzes signal-to-noise ratio with spike-aware baseline detection.

    Uses TransitionBasedSpikeDetector to identify:
    - Beginning baseline (before signal)
    - Laser spike artifact region
    - Signal decay region
    - Tail baseline (end of dataset)

    SNR is calculated using Mode 2 method:
        SNR = (peak_signal - baseline) / noise_std

    Where baseline and noise_std come from the tail baseline region
    for better statistical stability.
    """

    def __init__(self, spike_detector: Optional[TransitionBasedSpikeDetector] = None):
        """
        Initialize SNR analyzer.

        Args:
            spike_detector: Optional spike detector instance. If None, creates new one.
        """
        self.spike_detector = spike_detector or TransitionBasedSpikeDetector()

    def analyze_snr(self, x_data: np.ndarray, y_data: np.ndarray, dataset_type: str = 'auto') -> SNRResult:
        """
        Analyze SNR using tail baseline for better statistics.

        Workflow:
        1. Detect spikes and baselines using TransitionBasedSpikeDetector
        2. Extract signal region (after spike, before tail baseline)
        3. Calculate SNR = (peak - baseline) / noise_std
        4. Assess quality (Excellent/Good/Fair/Poor)

        Args:
            x_data: Time axis (μs)
            y_data: Intensity data (photon counts)
            dataset_type: Type of dataset for spike detection ('lag_spike', 'spike_only', 'clean_signal', 'preprocessed', 'auto')

        Returns:
            SNRResult dataclass with all metrics and baseline information
        """
        # Step 1: Detect spikes and baselines
        spike_results = self.spike_detector.detect_spikes(x_data, y_data, dataset_type)
        spike_region = spike_results.get('spike_region')

        # Step 2: Determine signal region boundaries
        # For SNR, we want the signal AFTER the spike/artifact (not shifted like kinetics fitting)
        if spike_region is not None:
            # Signal starts right after spike ends
            signal_start_idx = spike_region['end'] + 1
        else:
            # No spike detected, use signal_start_idx from detector
            signal_start_idx = spike_results.get('signal_start_idx', 0)

        # Clamp to valid range
        signal_start_idx = min(signal_start_idx, len(y_data) - 1)

        # Extract signal region (everything after spike/artifact)
        signal_x = x_data[signal_start_idx:]
        signal_y = y_data[signal_start_idx:]

        # Filter out NaN values from signal (for preprocessed data)
        valid_mask = ~np.isnan(signal_y)
        signal_y_valid = signal_y[valid_mask]
        signal_x_valid = signal_x[valid_mask]

        # Step 3: Extract BOTH baselines (beginning and tail)
        beginning_baseline = self._extract_beginning_baseline(y_data, spike_results)
        tail_baseline = self._extract_tail_baseline(y_data, spike_results)

        # Use tail baseline for SNR calculation (better statistics)
        baseline_level = tail_baseline['mean']
        noise_std = tail_baseline['std']

        # Apply minimum noise floor to prevent division by zero
        # For photon counting data, tail baseline can be all zeros (std=0)
        # Use minimum noise floor of 0.5 photon counts (shot noise limit)
        MIN_NOISE_FLOOR = 0.5
        if noise_std < MIN_NOISE_FLOOR:
            noise_std_used = MIN_NOISE_FLOOR
            logger.info(f"Tail noise std ({noise_std:.3f}) below floor, using {MIN_NOISE_FLOOR}")
        else:
            noise_std_used = noise_std

        # Step 4: Calculate SNR (Mode 2 method)
        if len(signal_y_valid) > 0:
            peak_signal = np.max(signal_y_valid)
            snr_linear = (peak_signal - baseline_level) / noise_std_used
            snr_db = 20 * np.log10(snr_linear)

            # Quality assessment
            quality = self._assess_quality(snr_db)

            # Determine which baseline was used
            baseline_used = 'tail'

            logger.info(f"SNR Analysis: {snr_db:.1f} dB ({quality})")
            logger.info(f"  Peak: {peak_signal:.2f}, Baseline: {baseline_level:.2f}, Noise: {noise_std:.2f}")

            return SNRResult(
                snr_db=snr_db,
                snr_linear=snr_linear,
                quality=quality,
                spike_region=spike_region,
                beginning_baseline=beginning_baseline,
                tail_baseline=tail_baseline,
                baseline_used=baseline_used,
                peak_signal=peak_signal,
                baseline_level=baseline_level,
                noise_std=noise_std_used,  # Use the noise floor-corrected value
                signal_region={'x': signal_x_valid, 'y': signal_y_valid, 'start_idx': signal_start_idx}
            )
        else:
            # Insufficient data for SNR calculation (no valid signal points)
            logger.warning("SNR calculation failed: no valid signal data")
            return SNRResult(
                snr_db=0.0,
                snr_linear=0.0,
                quality='Poor',  # Changed from 'Unknown' to 'Poor'
                spike_region=spike_region,
                beginning_baseline=beginning_baseline,
                tail_baseline=tail_baseline,
                baseline_used='tail',
                peak_signal=0.0,
                baseline_level=baseline_level,
                noise_std=noise_std_used,
                signal_region={'x': signal_x_valid, 'y': signal_y_valid, 'start_idx': signal_start_idx}
            )

    def _extract_beginning_baseline(self, y_data: np.ndarray,
                                   spike_results: dict) -> dict:
        """
        Extract beginning baseline (used by spike detector).

        Args:
            y_data: Intensity data
            spike_results: Results from spike detector

        Returns:
            Dictionary with baseline statistics and indices
        """
        beginning_baseline = spike_results.get('baseline')

        if beginning_baseline:
            start_idx = 0
            end_idx = beginning_baseline['end_idx']
            data = y_data[start_idx:end_idx + 1]
            mean = beginning_baseline['mean']
            std = beginning_baseline['std']
            length = beginning_baseline['length']
        else:
            # Fallback: use first 3 points
            start_idx = 0
            end_idx = 2
            data = y_data[:3]
            # Use nanmean/nanstd for robustness with preprocessed data
            mean = np.nanmean(data)
            std = np.nanstd(data)
            length = 3

        return {
            'y': data,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'mean': mean,
            'std': std,
            'length': length
        }

    def _extract_tail_baseline(self, y_data: np.ndarray,
                              spike_results: dict) -> dict:
        """
        Extract tail baseline (used for SNR calculation).

        Args:
            y_data: Intensity data
            spike_results: Results from spike detector

        Returns:
            Dictionary with baseline statistics and indices
        """
        tail_baseline = spike_results.get('tail_baseline')

        if tail_baseline is None:
            # Fallback: use last 100 points or 25% of dataset
            tail_length = min(100, len(y_data) // 4)
            data = y_data[-tail_length:]
            start_idx = len(y_data) - tail_length
            end_idx = len(y_data) - 1
            # Use nanmean/nanstd for robustness with preprocessed data
            mean = np.nanmean(data)
            std = np.nanstd(data)
        else:
            # Use detected tail baseline
            start_idx = tail_baseline['start_idx']
            end_idx = tail_baseline['end_idx']
            data = y_data[start_idx:end_idx + 1]
            mean = tail_baseline['mean']
            std = tail_baseline['std']
            tail_length = len(data)

        return {
            'y': data,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'mean': mean,
            'std': std,
            'length': len(data)
        }

    def _assess_quality(self, snr_db: float) -> str:
        """
        Assess signal quality based on SNR in dB.

        Args:
            snr_db: Signal-to-noise ratio in decibels

        Returns:
            Quality string: 'Excellent', 'Good', 'Fair', or 'Poor'
        """
        if snr_db > 20:
            return "Excellent"
        elif snr_db > 10:
            return "Good"
        elif snr_db > 3:
            return "Fair"
        else:
            return "Poor"

    def get_recommendations(self, snr_result: SNRResult) -> list[str]:
        """
        Generate analysis recommendations based on SNR quality.

        Args:
            snr_result: SNRResult from analyze_snr()

        Returns:
            List of recommendation strings
        """
        recommendations = []

        snr_db = snr_result.snr_db

        # Fitting recommendations
        if snr_db > 15:
            recommendations.append("Multi-exponential fitting should work reliably")
        elif snr_db > 8:
            recommendations.append("Single and bi-exponential fitting recommended")
        elif snr_db > 3:
            recommendations.append("Focus on single exponential fitting")
        else:
            recommendations.append("Consider signal averaging or noise reduction")

        # Spike detection confidence
        if snr_result.spike_region:
            spike_confidence = snr_result.spike_region.get('confidence', 1.0)
            if spike_confidence < 0.7:
                recommendations.append("Spike detection confidence is low - verify manually")

        return recommendations
