#!/usr/bin/env python3
"""
Core Fitting Module - Basic fitting functions and parameter handling
Split from kinetics_analyzer.py for better maintainability
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from utils.logger_config import get_logger

logger = get_logger(__name__)

def r2_score(y_true, y_pred):
    """
    Robust R² calculation with numerical stability checks.
    
    R² = 1 - (SS_res / SS_tot)
    where:
    SS_res = Σ(y_actual - y_predicted)²
    SS_tot = Σ(y_actual - y_mean)²
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check for valid input
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) < 2:
        raise ValueError("Need at least 2 data points for R² calculation")
    
    # Calculate residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate total sum of squares
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    # Handle edge cases
    if ss_tot == 0.0:
        # All y_true values are identical - perfect prediction should give R² = 1
        if ss_res == 0.0:
            return 1.0
        else:
            # Model predicts variation where none exists - very bad fit
            return 0.0
    
    # Check for numerical instability
    if ss_tot < 1e-15:
        logger.warning(f"Very small ss_tot ({ss_tot:.2e}) may cause numerical instability")
        logger.warning(f"ss_res: {ss_res:.2e}, ratio: {ss_res/ss_tot:.6f}")
    
    # Calculate R²
    r2 = 1.0 - (ss_res / ss_tot)
    
    # Clamp to reasonable range (R² can theoretically be negative for very bad fits)
    # But values > 1.0 indicate calculation errors
    if r2 > 1.0:
        logger.warning(f"R² > 1.0 ({r2:.6f}) indicates calculation error")
        logger.warning(f"ss_res: {ss_res:.2e}, ss_tot: {ss_tot:.2e}")
        r2 = 1.0
    
    return r2

class CoreFittingMethods:
    def __init__(self, tau_delta_default=3.5):
        """Initialize core fitting methods."""
        self.tau_delta_default = tau_delta_default
        self.fitted_t0 = None
    
    def load_data(self, filename):
        """Load CSV data and extract time and intensity columns."""
        try:
            data = pd.read_csv(filename, header=None)
            time = data.iloc[:, 0].values
            intensity = data.iloc[:, 1].values
            return time, intensity
        except Exception as e:
            raise ValueError(f"Error loading {filename}: {e}")
    
    def literature_biexponential_free(self, t, A, tau_delta, tau_T, t0, y0):
        """Main biexponential function f(t-t0) with all parameters free."""
        if abs(tau_delta - tau_T) < 1e-10:
            tau_T = tau_delta * 0.999
        
        t_shifted = np.maximum(t - t0, 0)
        factor = tau_delta / (tau_delta - tau_T)
        exp_term = np.exp(-t_shifted / tau_delta) - np.exp(-t_shifted / tau_T)
        return A * factor * exp_term + y0
    
    def literature_biexponential_pure(self, t, A, tau_delta, tau_T, y0):
        """Literature biexponential function f(t) without time offset."""
        if abs(tau_delta - tau_T) < 1e-10:
            tau_T = tau_delta * 0.999
        
        factor = tau_delta / (tau_delta - tau_T)
        exp_term = np.exp(-t / tau_delta) - np.exp(-t / tau_T)
        return A * factor * exp_term + y0

    def literature_biexponential_pure_fixed_tau_delta(self, t, A, tau_T, y0, tau_delta_fixed):
        """Literature biexponential function with τΔ fixed."""
        if abs(tau_delta_fixed - tau_T) < 1e-10:
            tau_T = tau_delta_fixed * 0.999
        
        factor = tau_delta_fixed / (tau_delta_fixed - tau_T)
        exp_term = np.exp(-t / tau_delta_fixed) - np.exp(-t / tau_T)
        return A * factor * exp_term + y0
    
    def detect_and_correct_parameter_exchange(self, A, tau_delta, tau_T, y0, expected_tau_delta):
        """Detect if parameters were exchanged and correct them."""
        corrections_applied = {'exchange': False, 'sign': False}
        
        # Step 1: Sign correction
        if A < 0:
            A = abs(A)
            corrections_applied['sign'] = True
        
        tolerance = max(0.5, expected_tau_delta * 0.3)
        
        # Step 2: Parameter exchange detection
        if abs(tau_delta - expected_tau_delta) <= tolerance:
            return A, tau_delta, tau_T, y0, corrections_applied
        
        if abs(tau_T - expected_tau_delta) <= tolerance:
            tau_delta_corrected = tau_T
            tau_T_corrected = tau_delta
            
            if abs(tau_delta_corrected - tau_T_corrected) > 1e-10:
                original_factor = tau_delta / (tau_delta - tau_T) if abs(tau_delta - tau_T) > 1e-10 else 1.0
                new_factor = tau_delta_corrected / (tau_delta_corrected - tau_T_corrected)
                A_corrected = A * original_factor / new_factor
                
                if A_corrected < 0:
                    A_corrected = abs(A_corrected)
                    corrections_applied['sign'] = True
            else:
                A_corrected = A
            
            corrections_applied['exchange'] = True
            return A_corrected, tau_delta_corrected, tau_T_corrected, y0, corrections_applied
        
        return A, tau_delta, tau_T, y0, corrections_applied
    
    def calculate_weighted_residuals(self, residuals, intensities):
        """Simple inverse variance weighting by intensity."""
        weights = np.sqrt(np.maximum(np.abs(intensities), 1.0))
        weighted_residuals = residuals / weights
        return weighted_residuals
    
    def calculate_auc(self, A, tau_delta, tau_T, y0=0):
        """
        Calculate the total area under the curve (AUC) from t0 to infinity.
        For the literature equation: I(t) = A × (τΔ/(τΔ-τT)) × (e^(-(t-t0)/τΔ) - e^(-(t-t0)/τT)) + y0
        The integral from t0 to infinity is: AUC = A × τΔ
        """
        return A * tau_delta
    
    def calculate_quantum_yield(self, A_sample, A_ref, abs_sample, abs_ref, qy_ref, 
                              AUC_sample=None, AUC_ref=None):
        """Calculate quantum yield using the relative method with both A and AUC."""
        abs_correction = (1 - 10**(-abs_ref)) / (1 - 10**(-abs_sample))
        
        qy_from_A = qy_ref * (A_sample / A_ref) * abs_correction
        
        qy_from_AUC = None
        if AUC_sample is not None and AUC_ref is not None:
            qy_from_AUC = qy_ref * (AUC_sample / AUC_ref) * abs_correction
        
        return qy_from_A, qy_from_AUC