#!/usr/bin/env python3
"""
Statistical Analyzer - Calculate means and standard deviations from kinetics_analyzer results
FIXED: Compatible with new kinetics_analyzer direct access structure
ENHANCED: Added chi-square statistics calculation
REFACTORED: Works with KineticsResult dataclasses (v2.5 Phase 2)
Following exact variable naming requirements for compatibility with other modules.
"""

import numpy as np
from utils.logger_config import get_logger
from core.kinetics_dataclasses import KineticsResult

logger = get_logger(__name__)


class StatisticalAnalyzer:
    """Calculate statistics from kinetics_analyzer replicate results."""
    
    def __init__(self):
        self.statistics = {}
        
        # Store mean and standard deviation arrays for other modules
        # EXPERIMENTAL METHOD (t timeline)
        self.mean_time_experiment_us = None
        self.sd_time_experiment_us = None
        self.mean_intensity_raw = None
        self.sd_intensity_raw = None
        self.mean_main_curve_ft_t0 = None
        self.sd_main_curve_ft_t0 = None
        self.mean_main_residuals = None
        self.sd_main_residuals = None
        self.mean_main_weighted_residuals = None
        self.sd_main_weighted_residuals = None
        self.mean_literature_curve_ft = None
        self.sd_literature_curve_ft = None
        self.mean_literature_residuals = None
        self.sd_literature_residuals = None
        self.mean_literature_weighted_residuals = None
        self.sd_literature_weighted_residuals = None

        # Mask information (use mask with most signal points)
        self.best_fitting_mask = None

        # STEP 4 METHOD (t' timeline)
        self.mean_time_t0_corrected_us = None
        self.sd_time_t0_corrected_us = None
        self.mean_intensity_raw_t_prime = None
        self.sd_intensity_raw_t_prime = None
        self.mean_step4_fitted_prediction = None
        self.sd_step4_fitted_prediction = None
        self.mean_step4_fitting_residuals = None
        self.sd_step4_fitting_residuals = None
        self.mean_step4_fitting_weighted_residuals = None
        self.sd_step4_fitting_weighted_residuals = None
        self.mean_step4_curve_ft_prime = None
        self.sd_step4_curve_ft_prime = None
    
    def extract_array_from_replicate(self, replicate_result, variable_name):
        """Extract a specific array variable from kinetics_analyzer stored results."""
        try:
            # PRIMARY: Handle KineticsResult dataclass (v2.5+)
            if isinstance(replicate_result, KineticsResult):
                # Map variable names to dataclass attributes
                if variable_name == 'time_experiment_us':
                    return replicate_result.time_experiment_us
                elif variable_name == 'intensity_raw':
                    return replicate_result.intensity_raw
                elif variable_name in ('main_curve_ft_t0', 'main_curve'):
                    return replicate_result.main_curve
                elif variable_name == 'main_weighted_residuals':
                    return replicate_result.main_weighted_residuals
                elif variable_name in ('literature_curve_ft', 'literature_curve'):
                    return replicate_result.literature.curve
                elif variable_name == 'literature_weighted_residuals':
                    return replicate_result.literature.weighted_residuals
                # Deprecated Step 4 fields - return None
                elif variable_name.startswith('step4_') or variable_name in ('time_t0_corrected_us', 'intensity_raw_t_prime'):
                    return None
                # Try direct attribute access as fallback
                elif hasattr(replicate_result, variable_name):
                    attr_value = getattr(replicate_result, variable_name)
                    if isinstance(attr_value, np.ndarray):
                        return attr_value

            # LEGACY: Handle dict-based results (for backward compatibility)
            elif isinstance(replicate_result, dict):
                # First, try direct access (old kinetics analyzer structure)
                if variable_name in replicate_result and replicate_result[variable_name] is not None:
                    array = replicate_result[variable_name]
                    # Convert to numpy array and ensure 1D
                    if isinstance(array, (list, np.ndarray)):
                        return np.array(array).flatten()

                # For backward compatibility: try accessing from model_curves
                model_curves = replicate_result.get('model_curves', {})
                if variable_name in model_curves and model_curves[variable_name] is not None:
                    array = model_curves[variable_name]
                    if isinstance(array, (list, np.ndarray)):
                        return np.array(array).flatten()

            # Fallback: Check if kinetics_analyzer has stored results (very old legacy support)
            if hasattr(replicate_result, '_stored_results') and replicate_result._stored_results is not None:
                stored = replicate_result._stored_results
                if variable_name in stored and stored[variable_name] is not None:
                    array = stored[variable_name]
                    if isinstance(array, (list, np.ndarray)):
                        return np.array(array).flatten()

            return None

        except Exception as e:
            logger.warning(f"Error extracting {variable_name}: {e}")
            return None
    
    def calculate_array_statistics(self, replicate_results, variable_name):
        """Calculate mean and std for a specific array variable across replicates.

        OPTIMIZED VERSION: Pre-allocates arrays and uses direct indexing instead of
        list comprehensions and multiple passes.
        """
        # Extract arrays from all replicates (single pass)
        valid_arrays = []
        for replicate in replicate_results:
            array = self.extract_array_from_replicate(replicate, variable_name)
            if array is not None and len(array) > 0:
                valid_arrays.append(array)

        if not valid_arrays:
            return None, None

        # Find minimum length (single pass through lengths)
        min_length = len(valid_arrays[0])
        max_length = min_length

        for arr in valid_arrays[1:]:
            arr_len = len(arr)
            if arr_len < min_length:
                min_length = arr_len
            if arr_len > max_length:
                max_length = arr_len

        if min_length == 0:
            return None, None

        # Warn if arrays have different lengths
        if min_length != max_length:
            logger.info(f"{variable_name} arrays have varying lengths ({min_length}-{max_length}). Truncating to {min_length}.")

        try:
            # Pre-allocate output array for stacking (avoids multiple copies)
            n_arrays = len(valid_arrays)
            stacked = np.empty((n_arrays, min_length), dtype=valid_arrays[0].dtype)

            # Fill stacked array directly (no intermediate list)
            for i, arr in enumerate(valid_arrays):
                stacked[i] = arr[:min_length]

            # Calculate mean and std along replicate axis (axis=0)
            with np.errstate(invalid='ignore'):
                mean_array = np.nanmean(stacked, axis=0)
                std_array = np.nanstd(stacked, axis=0, ddof=1)

            return mean_array, std_array

        except ValueError as e:
            logger.error(f"Error stacking arrays for {variable_name}: {e}")
            logger.error(f"  Array count: {len(valid_arrays)}, min_length: {min_length}")
            return None, None
    
    def calculate_statistics(self, replicate_results):
        """Calculate all statistics from kinetics_analyzer replicate results."""
        if not replicate_results:
            return {}
        
        logger.info(f"Calculating statistics for {len(replicate_results)} replicates")

        # Select best fitting mask (one with most signal points = most True values)
        best_mask = None
        max_signal_points = 0
        for result in replicate_results:
            if isinstance(result, KineticsResult) and hasattr(result, 'fitting_mask') and result.fitting_mask is not None:
                signal_points = np.sum(result.fitting_mask)  # Count True values
                if signal_points > max_signal_points:
                    max_signal_points = signal_points
                    best_mask = result.fitting_mask
        self.best_fitting_mask = best_mask
        if best_mask is not None:
            logger.info(f"Selected mask with {max_signal_points} signal points (out of {len(best_mask)})")

        # Calculate array statistics for EXPERIMENTAL METHOD (t timeline)
        self.mean_time_experiment_us, self.sd_time_experiment_us = self.calculate_array_statistics(
            replicate_results, 'time_experiment_us'
        )
        
        self.mean_intensity_raw, self.sd_intensity_raw = self.calculate_array_statistics(
            replicate_results, 'intensity_raw'
        )
        
        self.mean_main_curve_ft_t0, self.sd_main_curve_ft_t0 = self.calculate_array_statistics(
            replicate_results, 'main_curve_ft_t0'
        )
        
        self.mean_main_residuals, self.sd_main_residuals = self.calculate_array_statistics(
            replicate_results, 'main_residuals'
        )
        
        self.mean_main_weighted_residuals, self.sd_main_weighted_residuals = self.calculate_array_statistics(
            replicate_results, 'main_weighted_residuals'
        )
        
        self.mean_literature_curve_ft, self.sd_literature_curve_ft = self.calculate_array_statistics(
            replicate_results, 'literature_curve_ft'
        )
        
        self.mean_literature_residuals, self.sd_literature_residuals = self.calculate_array_statistics(
            replicate_results, 'literature_residuals'
        )
        
        self.mean_literature_weighted_residuals, self.sd_literature_weighted_residuals = self.calculate_array_statistics(
            replicate_results, 'literature_weighted_residuals'
        )
        
        # Calculate array statistics for STEP 4 METHOD (t' timeline)
        self.mean_time_t0_corrected_us, self.sd_time_t0_corrected_us = self.calculate_array_statistics(
            replicate_results, 'time_t0_corrected_us'
        )
        
        self.mean_intensity_raw_t_prime, self.sd_intensity_raw_t_prime = self.calculate_array_statistics(
            replicate_results, 'intensity_raw_t_prime'
        )
        
        self.mean_step4_fitted_prediction, self.sd_step4_fitted_prediction = self.calculate_array_statistics(
            replicate_results, 'step4_fitted_prediction'
        )
        
        self.mean_step4_fitting_residuals, self.sd_step4_fitting_residuals = self.calculate_array_statistics(
            replicate_results, 'step4_fitting_residuals'
        )
        
        self.mean_step4_fitting_weighted_residuals, self.sd_step4_fitting_weighted_residuals = self.calculate_array_statistics(
            replicate_results, 'step4_fitting_weighted_residuals'
        )
        
        self.mean_step4_curve_ft_prime, self.sd_step4_curve_ft_prime = self.calculate_array_statistics(
            replicate_results, 'step4_curve_ft_prime'
        )
        
        # Verify array length consistency
        self._verify_array_lengths()

        # NOTE: Global array consistency enforcement removed (Step 4 deprecated)
        # All Step 4 arrays are now None, no cross-timeline conflicts possible

        # Calculate parameter statistics
        parameter_stats = self.calculate_parameter_statistics(replicate_results)
        
        # Store all statistics
        self.statistics = parameter_stats
        
        logger.info("Statistics calculation complete")
        
        return self.statistics
    
    def _verify_array_lengths(self):
        """Verify that mean and SD arrays have matching lengths."""
        array_pairs = [
            ('mean_time_experiment_us', 'sd_time_experiment_us'),
            ('mean_intensity_raw', 'sd_intensity_raw'),
            ('mean_main_curve_ft_t0', 'sd_main_curve_ft_t0'),
            ('mean_main_residuals', 'sd_main_residuals'),
            ('mean_main_weighted_residuals', 'sd_main_weighted_residuals'),
            ('mean_literature_curve_ft', 'sd_literature_curve_ft'),
            ('mean_literature_residuals', 'sd_literature_residuals'),
            ('mean_literature_weighted_residuals', 'sd_literature_weighted_residuals'),
            ('mean_time_t0_corrected_us', 'sd_time_t0_corrected_us'),
            ('mean_intensity_raw_t_prime', 'sd_intensity_raw_t_prime'),
            ('mean_step4_fitted_prediction', 'sd_step4_fitted_prediction'),
            ('mean_step4_fitting_residuals', 'sd_step4_fitting_residuals'),
            ('mean_step4_fitting_weighted_residuals', 'sd_step4_fitting_weighted_residuals'),
            ('mean_step4_curve_ft_prime', 'sd_step4_curve_ft_prime'),
        ]
        
        mismatches = []
        for mean_name, sd_name in array_pairs:
            mean_arr = getattr(self, mean_name)
            sd_arr = getattr(self, sd_name)
            
            if mean_arr is not None and sd_arr is not None:
                if len(mean_arr) != len(sd_arr):
                    mismatches.append((mean_name, len(mean_arr), len(sd_arr)))
                    # Fix by truncating to minimum length
                    min_len = min(len(mean_arr), len(sd_arr))
                    setattr(self, mean_name, mean_arr[:min_len])
                    setattr(self, sd_name, sd_arr[:min_len])
        
        if mismatches:
            logger.warning("Array length mismatches detected and fixed")
            for name, mean_len, sd_len in mismatches:
                logger.warning(f"  {name}: mean={mean_len}, sd={sd_len} -> truncated to {min(mean_len, sd_len)}")
    
    # REMOVED: _enforce_global_array_consistency() method
    # No longer needed - Step 4 deprecated, all Step 4 arrays are None
    # No cross-timeline array length conflicts possible

    def extract_parameters_from_new_structure(self, replicate):
        """
        Extract parameters from KineticsResult dataclass or legacy dict structure.
        Includes chi-square parameters.
        """
        extracted_params = {}

        try:
            # PRIMARY: Handle KineticsResult dataclass (v2.5+)
            if isinstance(replicate, KineticsResult):
                # Extract parameters from FitParameters dataclass
                for param in ['A', 'tau_delta', 'tau_T', 't0', 'y0']:
                    val = getattr(replicate.parameters, param)
                    if val is not None and val != 'ND':
                        try:
                            extracted_params[param] = float(val)
                        except (ValueError, TypeError):
                            continue

                # Extract quality metrics from FitQuality dataclass
                for metric in ['r_squared', 'chi_square', 'reduced_chi_square']:
                    val = getattr(replicate.fit_quality, metric)
                    if val is not None:
                        try:
                            metric_float = float(val)
                            if not np.isnan(metric_float) and not np.isinf(metric_float):
                                extracted_params[metric] = metric_float
                        except (ValueError, TypeError):
                            pass

                # Phase 3A: Extract SNR metrics from SNRResult dataclass
                if replicate.snr_result is not None:
                    for metric in ['snr_db', 'snr_linear']:
                        val = getattr(replicate.snr_result, metric)
                        if val is not None:
                            try:
                                metric_float = float(val)
                                if not np.isnan(metric_float) and not np.isinf(metric_float):
                                    extracted_params[metric] = metric_float
                            except (ValueError, TypeError):
                                pass

                return extracted_params

            # LEGACY: Handle dict-based results
            elif isinstance(replicate, dict):
                # Direct parameter access (primary method)
                for param in ['A', 'tau_delta', 'tau_T', 't0', 'y0']:
                    if param in replicate:
                        val = replicate[param]
                        if val is not None and val != 'ND':
                            try:
                                extracted_params[param] = float(val)
                            except (ValueError, TypeError):
                                continue

                # R-squared
                if 'r_squared' in replicate:
                    val = replicate['r_squared']
                    if val is not None and val != 'ND':
                        try:
                            extracted_params['r_squared'] = float(val)
                        except (ValueError, TypeError):
                            pass

                # Chi-square
                if 'chi_square' in replicate:
                    val = replicate['chi_square']
                    if val is not None and val != 'ND':
                        try:
                            chi2_float = float(val)
                            if not np.isnan(chi2_float) and not np.isinf(chi2_float):
                                extracted_params['chi_square'] = chi2_float
                        except (ValueError, TypeError):
                            pass

                # Reduced chi-square
                if 'reduced_chi_square' in replicate:
                    val = replicate['reduced_chi_square']
                    if val is not None and val != 'ND':
                        try:
                            red_chi2_float = float(val)
                            if not np.isnan(red_chi2_float) and not np.isinf(red_chi2_float):
                                extracted_params['reduced_chi_square'] = red_chi2_float
                        except (ValueError, TypeError):
                            pass

                # If we got parameters, return them
                if extracted_params:
                    return extracted_params
                
                # FALLBACK: Try nested structure (backward compatibility)
                # Try main_model_result
                main_result = replicate.get('main_model_result')
                if main_result and main_result.get('success', False):
                    main_params = main_result.get('parameters', {})
                    for param in ['A', 'tau_delta', 'tau_T', 't0', 'y0']:
                        if param in main_params:
                            val = main_params[param]
                            if val is not None and val != 'ND':
                                try:
                                    extracted_params[param] = float(val)
                                except (ValueError, TypeError):
                                    continue
                    
                    # Get fit quality metrics
                    fit_quality = main_result.get('fit_quality', {})
                    
                    if 'r2' in fit_quality:
                        val = fit_quality['r2']
                        if val is not None and val != 'ND':
                            try:
                                extracted_params['r_squared'] = float(val)
                            except (ValueError, TypeError):
                                pass
                    
                    if 'chi_square' in fit_quality:
                        val = fit_quality['chi_square']
                        if val is not None and val != 'ND':
                            try:
                                chi2_float = float(val)
                                if not np.isnan(chi2_float) and not np.isinf(chi2_float):
                                    extracted_params['chi_square'] = chi2_float
                            except (ValueError, TypeError):
                                pass
                    
                    if 'reduced_chi_square' in fit_quality:
                        val = fit_quality['reduced_chi_square']
                        if val is not None and val != 'ND':
                            try:
                                red_chi2_float = float(val)
                                if not np.isnan(red_chi2_float) and not np.isinf(red_chi2_float):
                                    extracted_params['reduced_chi_square'] = red_chi2_float
                            except (ValueError, TypeError):
                                pass
                
                # Try literature_model_result if no main result
                if not extracted_params:
                    lit_result = replicate.get('literature_model_result')
                    if lit_result and lit_result.get('success', False):
                        for param in ['A', 'tau_delta', 'tau_T', 'y0']:
                            if param in lit_result:
                                val = lit_result[param]
                                if val is not None and val != 'ND':
                                    try:
                                        extracted_params[param] = float(val)
                                    except (ValueError, TypeError):
                                        continue
                        
                        if 'r_squared' in lit_result:
                            val = lit_result['r_squared']
                            if val is not None and val != 'ND':
                                try:
                                    extracted_params['r_squared'] = float(val)
                                except (ValueError, TypeError):
                                    pass
                        
                        if 'chi_square' in lit_result:
                            val = lit_result['chi_square']
                            if val is not None and val != 'ND':
                                try:
                                    chi2_float = float(val)
                                    if not np.isnan(chi2_float) and not np.isinf(chi2_float):
                                        extracted_params['chi_square'] = chi2_float
                                except (ValueError, TypeError):
                                    pass
                        
                        if 'reduced_chi_square' in lit_result:
                            val = lit_result['reduced_chi_square']
                            if val is not None and val != 'ND':
                                try:
                                    red_chi2_float = float(val)
                                    if not np.isnan(red_chi2_float) and not np.isinf(red_chi2_float):
                                        extracted_params['reduced_chi_square'] = red_chi2_float
                                except (ValueError, TypeError):
                                    pass
                
                # Try step4_t0_corrected_result if still no parameters
                if not extracted_params:
                    step4_result = replicate.get('step4_t0_corrected_result')
                    if step4_result and step4_result.get('success', False):
                        for param in ['A', 'tau_delta', 'tau_T', 'y0']:
                            if param in step4_result:
                                val = step4_result[param]
                                if val is not None and val != 'ND':
                                    try:
                                        extracted_params[param] = float(val)
                                    except (ValueError, TypeError):
                                        continue
                        
                        if 'r_squared' in step4_result:
                            val = step4_result['r_squared']
                            if val is not None and val != 'ND':
                                try:
                                    extracted_params['r_squared'] = float(val)
                                except (ValueError, TypeError):
                                    pass
                        
                        if 'chi_square' in step4_result:
                            val = step4_result['chi_square']
                            if val is not None and val != 'ND':
                                try:
                                    chi2_float = float(val)
                                    if not np.isnan(chi2_float) and not np.isinf(chi2_float):
                                        extracted_params['chi_square'] = chi2_float
                                except (ValueError, TypeError):
                                    pass
                        
                        if 'reduced_chi_square' in step4_result:
                            val = step4_result['reduced_chi_square']
                            if val is not None and val != 'ND':
                                try:
                                    red_chi2_float = float(val)
                                    if not np.isnan(red_chi2_float) and not np.isinf(red_chi2_float):
                                        extracted_params['reduced_chi_square'] = red_chi2_float
                                except (ValueError, TypeError):
                                    pass
            
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Could not extract parameters from replicate: {e}")
            
        return extracted_params
    
    def calculate_parameter_statistics(self, replicate_results):
        """
        Calculate statistics for kinetics parameters from new structure.
        Includes chi-square statistics.

        OPTIMIZED Phase 3: Vectorized parameter extraction with reduced type checking.
        """
        parameters = {
            'A': [],
            'tau_delta': [],
            'tau_T': [],
            't0': [],
            'y0': [],
            'r_squared': [],
            'chi_square': [],
            'reduced_chi_square': [],
            # Phase 3A: SNR parameters
            'snr_db': [],
            'snr_linear': []
        }

        # OPTIMIZED: Extract all parameters in one pass
        # Pre-allocate lists for better memory efficiency
        param_names = tuple(parameters.keys())  # Tuple is faster than dict.keys()

        for i, replicate in enumerate(replicate_results):
            try:
                extracted_params = self.extract_parameters_from_new_structure(replicate)

                # OPTIMIZED: Direct dict iteration without redundant lookups
                for param_name, param_value in extracted_params.items():
                    if param_name in parameters:  # Fast membership test
                        # OPTIMIZED: Combined validation in single check
                        try:
                            val_float = float(param_value)
                            # Use bitwise operation for NaN/inf check (faster than separate calls)
                            if np.isfinite(val_float):  # Combines isnan() and isinf() checks
                                parameters[param_name].append(val_float)
                        except (ValueError, TypeError):
                            pass  # Skip invalid values silently

            except (ValueError, TypeError, AttributeError, KeyError) as e:
                logger.warning(f"Could not extract parameters from replicate {i}: {e}")
                continue
        
        # Calculate statistics
        stats = {}
        
        logger.info("Parameter Statistics Summary:")
        for param_name, values in parameters.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

                stats[f'{param_name}_mean'] = mean_val
                stats[f'{param_name}_sd'] = std_val
                stats[f'{param_name}_n'] = len(values)

                logger.info(f"  {param_name}: {mean_val:.6f} ± {std_val:.6f} (n={len(values)})")
            else:
                stats[f'{param_name}_mean'] = 'ND'
                stats[f'{param_name}_sd'] = 'ND'
                stats[f'{param_name}_n'] = 0
                logger.info(f"  {param_name}: No data")

        # Verify chi-square consistency
        if stats['chi_square_n'] > 0 and stats['r_squared_n'] > 0:
            logger.info("Fit Quality Check:")
            logger.info(f"  R²: {stats['r_squared_mean']:.4f} (higher is better, max=1.0)")
            logger.info(f"  Chi²: {stats['chi_square_mean']:.4f} (lower is better)")
            logger.info(f"  Chi²_red: {stats['reduced_chi_square_mean']:.4f} (should be ~1.0 for good fit)")

            # Warning if values seem inconsistent
            if stats['r_squared_mean'] > 0.95 and stats['reduced_chi_square_mean'] > 10:
                logger.warning("High R² but large reduced chi-square - check error estimates!")
            elif stats['r_squared_mean'] < 0.80 and stats['reduced_chi_square_mean'] < 2:
                logger.warning("Low R² but small reduced chi-square - possible underestimated errors!")
        
        return stats
    
    def get_mean_arrays(self):
        """Return dictionary of all mean arrays for other modules."""
        return {
            # EXPERIMENTAL METHOD (t timeline)
            'mean_time_experiment_us': self.mean_time_experiment_us,
            'mean_intensity_raw': self.mean_intensity_raw,
            'mean_main_curve_ft_t0': self.mean_main_curve_ft_t0,
            'mean_main_residuals': self.mean_main_residuals,
            'mean_main_weighted_residuals': self.mean_main_weighted_residuals,
            'mean_literature_curve_ft': self.mean_literature_curve_ft,
            'mean_literature_residuals': self.mean_literature_residuals,
            'mean_literature_weighted_residuals': self.mean_literature_weighted_residuals,

            # Mask information (best mask with most signal points)
            'best_fitting_mask': self.best_fitting_mask,

            # STEP 4 METHOD (t' timeline)
            'mean_time_t0_corrected_us': self.mean_time_t0_corrected_us,
            'mean_intensity_raw_t_prime': self.mean_intensity_raw_t_prime,
            'mean_step4_fitted_prediction': self.mean_step4_fitted_prediction,
            'mean_step4_fitting_residuals': self.mean_step4_fitting_residuals,
            'mean_step4_fitting_weighted_residuals': self.mean_step4_fitting_weighted_residuals,
            'mean_step4_curve_ft_prime': self.mean_step4_curve_ft_prime,
        }
    
    def get_sd_arrays(self):
        """Return dictionary of all standard deviation arrays for other modules."""
        return {
            # EXPERIMENTAL METHOD (t timeline)
            'sd_time_experiment_us': self.sd_time_experiment_us,
            'sd_intensity_raw': self.sd_intensity_raw,
            'sd_main_curve_ft_t0': self.sd_main_curve_ft_t0,
            'sd_main_residuals': self.sd_main_residuals,
            'sd_main_weighted_residuals': self.sd_main_weighted_residuals,
            'sd_literature_curve_ft': self.sd_literature_curve_ft,
            'sd_literature_residuals': self.sd_literature_residuals,
            'sd_literature_weighted_residuals': self.sd_literature_weighted_residuals,
            
            # STEP 4 METHOD (t' timeline)
            'sd_time_t0_corrected_us': self.sd_time_t0_corrected_us,
            'sd_intensity_raw_t_prime': self.sd_intensity_raw_t_prime,
            'sd_step4_fitted_prediction': self.sd_step4_fitted_prediction,
            'sd_step4_fitting_residuals': self.sd_step4_fitting_residuals,
            'sd_step4_fitting_weighted_residuals': self.sd_step4_fitting_weighted_residuals,
            'sd_step4_curve_ft_prime': self.sd_step4_curve_ft_prime,
        }