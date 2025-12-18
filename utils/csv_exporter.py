#!/usr/bin/env python3
"""
CSV Exporter - Export individual replicate files and statistical summary files
REFACTORED: Works with KineticsResult dataclasses (v2.5 Phase 2)
Following exact variable naming requirements for compatibility with other modules.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from core.kinetics_dataclasses import KineticsResult
from utils.logger_config import get_logger

logger = get_logger(__name__)


class CSVExporter:
    """Export kinetics_analyzer results and statistical_analyzer statistics to CSV files."""

    def __init__(self, output_dir="exports/csv_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
                elif variable_name == 'main_residuals':
                    # Deprecated: unweighted residuals removed, return None
                    return None
                elif variable_name == 'main_weighted_residuals':
                    return replicate_result.main_weighted_residuals
                elif variable_name in ('literature_curve_ft', 'literature_curve'):
                    return replicate_result.literature.curve
                elif variable_name == 'literature_residuals':
                    # Deprecated: unweighted residuals removed, return None
                    return None
                elif variable_name == 'literature_weighted_residuals':
                    return replicate_result.literature.weighted_residuals
                elif variable_name == 'fitting_mask':
                    return replicate_result.fitting_mask.astype(int)
                elif variable_name == 'spike_region':
                    return replicate_result.spike_region.astype(int)
                # Deprecated Step 4 fields - return None
                elif variable_name.startswith('step4_') or variable_name in (
                    'time_t0_corrected_us', 'intensity_raw_t_prime', 't_prime_positive'
                ):
                    return None
                # Try direct attribute access as fallback
                elif hasattr(replicate_result, variable_name):
                    attr_value = getattr(replicate_result, variable_name)
                    if isinstance(attr_value, np.ndarray):
                        return attr_value
                return None

            # LEGACY: Handle dict-based results
            elif isinstance(replicate_result, dict):
                # Check direct dictionary access
                if variable_name in replicate_result and replicate_result[variable_name] is not None:
                    return np.array(replicate_result[variable_name])

                # Check for stored arrays in dictionary
                for key in [f'{variable_name}_full', f'{variable_name}']:
                    if key in replicate_result and replicate_result[key] is not None:
                        return np.array(replicate_result[key])

            # Check if kinetics_analyzer has stored results (very old legacy)
            if hasattr(replicate_result, '_stored_results') and replicate_result._stored_results is not None:
                stored = replicate_result._stored_results
                if variable_name in stored and stored[variable_name] is not None:
                    return np.array(stored[variable_name])

            return None

        except Exception:
            return None

    def calculate_step4_arrays(self, replicate_result):
        """Calculate step4_fitted_data_only, step4_fitted_prediction, etc. from stored results."""
        try:
            # Get stored results
            if not hasattr(replicate_result, '_stored_results') or replicate_result._stored_results is None:
                return {}

            stored = replicate_result._stored_results

            # Get basic arrays
            time_experiment = stored.get('time_experiment_us')
            fitting_mask = stored.get('fitting_mask')
            step4_result = stored.get('step4_result')

            if time_experiment is None or fitting_mask is None or not step4_result or not step4_result.get('success'):
                return {}

            # Initialize arrays with NaN
            step4_fitted_data_only = np.full(len(time_experiment), np.nan)
            step4_fitted_prediction = np.full(len(time_experiment), np.nan)
            step4_fitting_residuals = np.full(len(time_experiment), np.nan)
            step4_fitting_weighted_residuals = np.full(len(time_experiment), np.nan)

            # Get fitting data from step4_result
            if 'fitting_data' in step4_result:
                fitting_data = step4_result['fitting_data']

                t_fit_step4 = fitting_data.get('t_fit')
                I_fit_step4 = fitting_data.get('I_fit')
                I_pred_step4 = fitting_data.get('I_pred')

                if t_fit_step4 is not None and I_fit_step4 is not None and I_pred_step4 is not None:
                    # Map fitted data back to full arrays
                    fitting_indices = np.where(fitting_mask)[0]

                    if len(t_fit_step4) == len(fitting_indices):
                        step4_fitted_data_only[fitting_indices] = I_fit_step4
                        step4_fitted_prediction[fitting_indices] = I_pred_step4

                        # Calculate residuals
                        residuals = I_fit_step4 - I_pred_step4
                        step4_fitting_residuals[fitting_indices] = residuals

                        # Calculate weighted residuals
                        weights = np.sqrt(np.maximum(np.abs(I_fit_step4), 1.0))
                        step4_fitting_weighted_residuals[fitting_indices] = residuals / weights

            # Calculate t_prime_positive
            time_t0_corrected = stored.get('time_t0_corrected_us')
            t_prime_positive = None
            if time_t0_corrected is not None:
                t_prime_positive = (time_t0_corrected >= 0).astype(int)

            return {
                'step4_fitted_data_only': step4_fitted_data_only,
                'step4_fitted_prediction': step4_fitted_prediction,
                'step4_fitting_residuals': step4_fitting_residuals,
                'step4_fitting_weighted_residuals': step4_fitting_weighted_residuals,
                't_prime_positive': t_prime_positive
            }

        except Exception:
            return {}

    def export_individual_replicate(self, compound_name, replicate_idx, replicate_result):
        """Export individual replicate CSV file with all required variables."""
        try:
            # Extract EXPERIMENTAL METHOD (t timeline) arrays
            time_experiment_us = self.extract_array_from_replicate(replicate_result, 'time_experiment_us')
            intensity_raw = self.extract_array_from_replicate(replicate_result, 'intensity_raw')
            main_curve_ft_t0 = self.extract_array_from_replicate(replicate_result, 'main_curve_ft_t0')
            main_residuals = self.extract_array_from_replicate(replicate_result, 'main_residuals')
            main_weighted_residuals = self.extract_array_from_replicate(replicate_result, 'main_weighted_residuals')
            literature_curve_ft = self.extract_array_from_replicate(replicate_result, 'literature_curve_ft')
            literature_residuals = self.extract_array_from_replicate(replicate_result, 'literature_residuals')
            literature_weighted_residuals = self.extract_array_from_replicate(replicate_result, 'literature_weighted_residuals')
            fitting_mask = self.extract_array_from_replicate(replicate_result, 'fitting_mask')
            spike_region = self.extract_array_from_replicate(replicate_result, 'spike_region')

            # Extract STEP 4 METHOD (t' timeline) arrays
            time_t0_corrected_us = self.extract_array_from_replicate(replicate_result, 'time_t0_corrected_us')
            intensity_raw_t_prime = self.extract_array_from_replicate(replicate_result, 'intensity_raw_t_prime')
            step4_curve_ft_prime = self.extract_array_from_replicate(replicate_result, 'step4_curve_ft_prime')
            step4_full_residuals = self.extract_array_from_replicate(replicate_result, 'step4_full_residuals')
            step4_weighted_residuals = self.extract_array_from_replicate(replicate_result, 'step4_weighted_residuals')

            # Calculate step4 fitted arrays
            step4_arrays = self.calculate_step4_arrays(replicate_result)
            step4_fitted_data_only = step4_arrays.get('step4_fitted_data_only')
            step4_fitted_prediction = step4_arrays.get('step4_fitted_prediction')
            step4_fitting_residuals = step4_arrays.get('step4_fitting_residuals')
            step4_fitting_weighted_residuals = step4_arrays.get('step4_fitting_weighted_residuals')
            t_prime_positive = step4_arrays.get('t_prime_positive')

            # Check if we have minimum required data
            if time_experiment_us is None or intensity_raw is None:
                return None

            # Create data dictionary
            export_data = {}

            # EXPERIMENTAL METHOD (t timeline)
            if time_experiment_us is not None:
                export_data['time_experiment_us'] = time_experiment_us
            if intensity_raw is not None:
                export_data['intensity_raw'] = intensity_raw
            if main_curve_ft_t0 is not None:
                export_data['main_curve_ft_t0'] = main_curve_ft_t0
            if main_residuals is not None:
                export_data['main_residuals'] = main_residuals
            if main_weighted_residuals is not None:
                export_data['main_weighted_residuals'] = main_weighted_residuals
            if literature_curve_ft is not None:
                export_data['literature_curve_ft'] = literature_curve_ft
            if literature_residuals is not None:
                export_data['literature_residuals'] = literature_residuals
            if literature_weighted_residuals is not None:
                export_data['literature_weighted_residuals'] = literature_weighted_residuals
            if fitting_mask is not None:
                export_data['fitting_mask'] = fitting_mask
            if spike_region is not None:
                export_data['spike_region'] = spike_region

            # STEP 4 METHOD (t' timeline)
            if time_t0_corrected_us is not None:
                export_data['time_t0_corrected_us'] = time_t0_corrected_us
            if intensity_raw_t_prime is not None:
                export_data['intensity_raw_t_prime'] = intensity_raw_t_prime
            if step4_fitted_data_only is not None:
                export_data['step4_fitted_data_only'] = step4_fitted_data_only
            if step4_fitted_prediction is not None:
                export_data['step4_fitted_prediction'] = step4_fitted_prediction
            if step4_fitting_residuals is not None:
                export_data['step4_fitting_residuals'] = step4_fitting_residuals
            if step4_fitting_weighted_residuals is not None:
                export_data['step4_fitting_weighted_residuals'] = step4_fitting_weighted_residuals
            if step4_curve_ft_prime is not None:
                export_data['step4_curve_ft_prime'] = step4_curve_ft_prime
            if step4_full_residuals is not None:
                export_data['step4_full_residuals'] = step4_full_residuals
            if step4_weighted_residuals is not None:
                export_data['step4_weighted_residuals'] = step4_weighted_residuals
            if t_prime_positive is not None:
                export_data['t_prime_positive'] = t_prime_positive

            # Create DataFrame and export
            df = pd.DataFrame(export_data)
            filename = self.output_dir / f"{compound_name}_rep{replicate_idx:02d}_data.csv"
            df.to_csv(filename, index=False, float_format='%.8f')

            return str(filename)

        except Exception as e:
            logger.error(f"Error exporting replicate {replicate_idx} for {compound_name}: {e}")
            return None

    def export_statistical_summary(self, compound_name, statistical_analyzer):
        """Export statistical summary CSV file with means and standard deviations."""
        try:
            # Get mean and sd arrays from statistical_analyzer
            mean_arrays = statistical_analyzer.get_mean_arrays()
            sd_arrays = statistical_analyzer.get_sd_arrays()

            # Create data dictionary
            export_data = {}

            # EXPERIMENTAL METHOD (t timeline) - means and standard deviations
            if mean_arrays['mean_time_experiment_us'] is not None:
                export_data['mean_time_experiment_us'] = mean_arrays['mean_time_experiment_us']
            if sd_arrays['sd_time_experiment_us'] is not None:
                export_data['sd_time_experiment_us'] = sd_arrays['sd_time_experiment_us']

            if mean_arrays['mean_intensity_raw'] is not None:
                export_data['mean_intensity_raw'] = mean_arrays['mean_intensity_raw']
            if sd_arrays['sd_intensity_raw'] is not None:
                export_data['sd_intensity_raw'] = sd_arrays['sd_intensity_raw']

            if mean_arrays['mean_main_curve_ft_t0'] is not None:
                export_data['mean_main_curve_ft_t0'] = mean_arrays['mean_main_curve_ft_t0']
            if sd_arrays['sd_main_curve_ft_t0'] is not None:
                export_data['sd_main_curve_ft_t0'] = sd_arrays['sd_main_curve_ft_t0']

            if mean_arrays['mean_main_residuals'] is not None:
                export_data['mean_main_residuals'] = mean_arrays['mean_main_residuals']
            if sd_arrays['sd_main_residuals'] is not None:
                export_data['sd_main_residuals'] = sd_arrays['sd_main_residuals']

            if mean_arrays['mean_main_weighted_residuals'] is not None:
                export_data['mean_main_weighted_residuals'] = mean_arrays['mean_main_weighted_residuals']
            if sd_arrays['sd_main_weighted_residuals'] is not None:
                export_data['sd_main_weighted_residuals'] = sd_arrays['sd_main_weighted_residuals']

            if mean_arrays['mean_literature_curve_ft'] is not None:
                export_data['mean_literature_curve_ft'] = mean_arrays['mean_literature_curve_ft']
            if sd_arrays['sd_literature_curve_ft'] is not None:
                export_data['sd_literature_curve_ft'] = sd_arrays['sd_literature_curve_ft']

            if mean_arrays['mean_literature_residuals'] is not None:
                export_data['mean_literature_residuals'] = mean_arrays['mean_literature_residuals']
            if sd_arrays['sd_literature_residuals'] is not None:
                export_data['sd_literature_residuals'] = sd_arrays['sd_literature_residuals']

            if mean_arrays['mean_literature_weighted_residuals'] is not None:
                export_data['mean_literature_weighted_residuals'] = mean_arrays['mean_literature_weighted_residuals']
            if sd_arrays['sd_literature_weighted_residuals'] is not None:
                export_data['sd_literature_weighted_residuals'] = sd_arrays['sd_literature_weighted_residuals']

            # STEP 4 METHOD (t' timeline) - means and standard deviations
            if mean_arrays['mean_time_t0_corrected_us'] is not None:
                export_data['mean_time_t0_corrected_us'] = mean_arrays['mean_time_t0_corrected_us']
            if sd_arrays['sd_time_t0_corrected_us'] is not None:
                export_data['sd_time_t0_corrected_us'] = sd_arrays['sd_time_t0_corrected_us']

            if mean_arrays['mean_intensity_raw_t_prime'] is not None:
                export_data['mean_intensity_raw_t_prime'] = mean_arrays['mean_intensity_raw_t_prime']
            if sd_arrays['sd_intensity_raw_t_prime'] is not None:
                export_data['sd_intensity_raw_t_prime'] = sd_arrays['sd_intensity_raw_t_prime']

            if mean_arrays['mean_step4_fitted_prediction'] is not None:
                export_data['mean_step4_fitted_prediction'] = mean_arrays['mean_step4_fitted_prediction']
            if sd_arrays['sd_step4_fitted_prediction'] is not None:
                export_data['sd_step4_fitted_prediction'] = sd_arrays['sd_step4_fitted_prediction']

            if mean_arrays['mean_step4_fitting_residuals'] is not None:
                export_data['mean_step4_fitting_residuals'] = mean_arrays['mean_step4_fitting_residuals']
            if sd_arrays['sd_step4_fitting_residuals'] is not None:
                export_data['sd_step4_fitting_residuals'] = sd_arrays['sd_step4_fitting_residuals']

            if mean_arrays['mean_step4_fitting_weighted_residuals'] is not None:
                export_data['mean_step4_fitting_weighted_residuals'] = mean_arrays['mean_step4_fitting_weighted_residuals']
            if sd_arrays['sd_step4_fitting_weighted_residuals'] is not None:
                export_data['sd_step4_fitting_weighted_residuals'] = sd_arrays['sd_step4_fitting_weighted_residuals']

            if mean_arrays['mean_step4_curve_ft_prime'] is not None:
                export_data['mean_step4_curve_ft_prime'] = mean_arrays['mean_step4_curve_ft_prime']
            if sd_arrays['sd_step4_curve_ft_prime'] is not None:
                export_data['sd_step4_curve_ft_prime'] = sd_arrays['sd_step4_curve_ft_prime']

            # Check if we have any data to export
            if not export_data:
                return None

            # Create DataFrame and export
            df = pd.DataFrame(export_data)
            filename = self.output_dir / f"{compound_name}_means.csv"
            df.to_csv(filename, index=False, float_format='%.8f')

            return str(filename)

        except Exception as e:
            logger.error(f"Error exporting statistical summary for {compound_name}: {e}")
            return None

    def export_compound(self, compound_name, replicate_results, statistical_analyzer):
        """Export all files for a compound: individual replicates + statistical summary."""
        exported_files = {
            'individual_replicates': [],
            'statistical_summary': None
        }

        # Export individual replicates
        for idx, replicate_result in enumerate(replicate_results, 1):
            filename = self.export_individual_replicate(compound_name, idx, replicate_result)
            if filename:
                exported_files['individual_replicates'].append(filename)

        # Export statistical summary
        summary_filename = self.export_statistical_summary(compound_name, statistical_analyzer)
        if summary_filename:
            exported_files['statistical_summary'] = summary_filename

        return exported_files

    def export_individual_replicate_parameters(self, compound_name, replicate_results):
        """Export individual replicate parameters (one row per replicate) for a compound."""
        try:
            from core.kinetics_dataclasses import KineticsResult
            replicate_data = []

            for idx, result in enumerate(replicate_results, 1):
                if not isinstance(result, KineticsResult):
                    continue

                # Extract parameters from KineticsResult
                row = {
                    'compound': compound_name,
                    'replicate': idx,
                    'alpha': result.parameters.A,
                    'tau_delta': result.parameters.tau_delta,
                    'tau_T': result.parameters.tau_T if result.parameters.tau_T != 'ND' else None,
                    't0': result.parameters.t0,
                    'y0': result.parameters.y0,
                    'r_squared': result.fit_quality.r_squared,
                    'chi_square': result.fit_quality.chi_square,
                    'reduced_chi_square': result.fit_quality.reduced_chi_square,
                    'model_used': result.fit_quality.model_used,
                    'snr_db': result.snr_result.snr_db if result.snr_result else None,
                    'snr_linear': result.snr_result.snr_linear if result.snr_result else None,
                    'snr_quality': result.snr_result.quality if result.snr_result else None
                }

                replicate_data.append(row)

            if not replicate_data:
                return None

            # Create DataFrame and export
            df = pd.DataFrame(replicate_data)
            filename = self.output_dir / f"{compound_name}_replicate_parameters.csv"
            df.to_csv(filename, index=False, float_format='%.8f')

            return str(filename)

        except Exception as e:
            logger.error(f"Error exporting replicate parameters for {compound_name}: {e}")
            return None

    def export_parameter_statistics(self, analysis_results):
        """Export parameter statistics CSV file with mean/sd separated."""
        try:
            stats_data = []

            for compound, compound_list in analysis_results.items():
                for compound_data in compound_list:
                    if 'statistics' not in compound_data:
                        continue

                    stats = compound_data['statistics']
                    wavelength = compound_data.get('wavelength', 'unknown')
                    classification = compound_data.get('classification', 'unknown')

                    # Calculate masked_time_mean from mean_arrays if available
                    masked_time_mean = None
                    if 'mean_arrays' in compound_data:
                        mean_arrays = compound_data['mean_arrays']
                        if 'fitting_mask' in mean_arrays and 'time_experiment_us' in mean_arrays:
                            mask = mean_arrays['fitting_mask']
                            time = mean_arrays['time_experiment_us']
                            import numpy as np
                            masked_indices = np.where(mask)[0]
                            if len(masked_indices) > 0:
                                masked_time_mean = time[masked_indices[0]]

                    row = {
                        'compound': compound,
                        'classification': classification,
                        'wavelength': wavelength,
                        'n_replicates': stats.get('A_n', 0),
                        'alpha_mean': stats.get('A_mean'),  # A renamed to alpha in CSV headers
                        'alpha_sd': stats.get('A_sd'),
                        'tau_delta_mean': stats.get('tau_delta_mean'),
                        'tau_delta_sd': stats.get('tau_delta_sd'),
                        'tau_T_mean': stats.get('tau_T_mean') if stats.get('tau_T_mean') != 'ND' else None,
                        'tau_T_sd': stats.get('tau_T_sd') if stats.get('tau_T_sd') != 'ND' else None,
                        'masked_time_us_mean': masked_time_mean,
                        't0_mean': stats.get('t0_mean'),
                        't0_sd': stats.get('t0_sd'),
                        'y0_mean': stats.get('y0_mean'),
                        'y0_sd': stats.get('y0_sd'),
                        'r_squared_mean': stats.get('r_squared_mean'),
                        'r_squared_sd': stats.get('r_squared_sd'),
                        # Phase 3A: SNR columns
                        'snr_db_mean': stats.get('snr_db_mean'),
                        'snr_db_sd': stats.get('snr_db_sd'),
                        'snr_linear_mean': stats.get('snr_linear_mean'),
                        'snr_linear_sd': stats.get('snr_linear_sd')
                    }

                    stats_data.append(row)

            if not stats_data:
                return None

            # Create DataFrame and export
            df = pd.DataFrame(stats_data)
            filename = self.output_dir / "parameter_statistics.csv"
            df.to_csv(filename, index=False, float_format='%.8f')

            return str(filename)

        except Exception as e:
            logger.error(f"Error exporting parameter statistics: {e}")
            return None

    def export_quantum_yields(self, qy_results):
        """Export quantum yield results CSV file with comprehensive method results."""
        try:
            qy_data = []

            for qy in qy_results:
                if not qy.get('success', False):
                    continue

                # Basic QY data
                row = {
                    'sample_compound': qy.get('sample_compound'),
                    'standard_compound': qy.get('standard_compound'),
                    'wavelength': qy.get('wavelength'),
                    'n_calculations': qy.get('n_calculations'),
                    'primary_QY': qy.get('quantum_yield'),
                    'primary_QY_sd': qy.get('quantum_yield_error'),
                    'statistical_error': qy.get('statistical_error'),
                    'systematic_error': qy.get('systematic_error'),
                    'relative_error_percent': qy.get('relative_error_percent'),
                    'sample_model_type': qy.get('sample_model_type'),
                    'standard_model_type': qy.get('standard_model_type')
                }

                # Enhanced analysis methods if available
                if 'enhanced_analysis' in qy:
                    enhanced = qy['enhanced_analysis']
                    methods = enhanced.get('methods', {})

                    # A_based method
                    if 'A_based' in methods and methods['A_based'].get('success'):
                        a_method = methods['A_based']
                        row['A_based_QY'] = a_method.get('quantum_yield')
                        row['A_based_QY_sd'] = a_method.get('quantum_yield_error')
                        row['A_based_n_calc'] = a_method.get('n_calculations')

                    # S0_based method
                    if 'S0_based' in methods and methods['S0_based'].get('success'):
                        s0_method = methods['S0_based']
                        row['S0_based_QY'] = s0_method.get('quantum_yield')
                        row['S0_based_QY_sd'] = s0_method.get('quantum_yield_error')
                        row['S0_based_n_calc'] = s0_method.get('n_calculations')

                    # S0_t0_corrected_based method
                    if 'S0_t0_corrected_based' in methods and methods['S0_t0_corrected_based'].get('success'):
                        s0_t0_method = methods['S0_t0_corrected_based']
                        row['S0_t0_corrected_QY'] = s0_t0_method.get('quantum_yield')
                        row['S0_t0_corrected_QY_sd'] = s0_t0_method.get('quantum_yield_error')
                        row['S0_t0_corrected_n_calc'] = s0_t0_method.get('n_calculations')
                        row['S0_t0_corrected_source'] = s0_t0_method.get('amplitude_source', 'unknown')

                    # AUC_based method
                    if 'AUC_based' in methods and methods['AUC_based'].get('success'):
                        auc_method = methods['AUC_based']
                        row['AUC_based_QY'] = auc_method.get('quantum_yield')
                        row['AUC_based_QY_sd'] = auc_method.get('quantum_yield_error')
                        row['AUC_based_n_calc'] = auc_method.get('n_calculations')

                    # Method agreement analysis
                    comparison = enhanced.get('comparison', {})
                    if 'A_vs_S0_t0_corrected_agreement' in comparison:
                        agreement = comparison['A_vs_S0_t0_corrected_agreement']
                        row['A_vs_S0_t0_agreement_percent'] = agreement.get('relative_difference_percent')
                        row['A_vs_S0_t0_agreement_level'] = agreement.get('agreement_level')

                    if 'A_vs_S0_agreement' in comparison:
                        agreement = comparison['A_vs_S0_agreement']
                        row['A_vs_S0_agreement_percent'] = agreement.get('relative_difference_percent')
                        row['A_vs_S0_agreement_level'] = agreement.get('agreement_level')

                    if 'cross_method_variability' in comparison:
                        variability = comparison['cross_method_variability']
                        row['cross_method_cv_percent'] = variability.get('cv_percent')

                qy_data.append(row)

            if not qy_data:
                return None

            # Create DataFrame and export
            df = pd.DataFrame(qy_data)
            filename = self.output_dir / "quantum_yields.csv"
            df.to_csv(filename, index=False, float_format='%.8f')

            return str(filename)

        except Exception as e:
            logger.error(f"Error exporting quantum yields: {e}")
            return None

    def export_all_compounds(self, compounds_data, analysis_results=None, qy_results=None):
        """Export all compounds data including parameter statistics and QY results.

        Args:
            compounds_data: Dictionary with replicate_results and statistical_analyzer
            analysis_results: Dictionary with statistics (optional, for parameter stats)
            qy_results: List of QY results (optional, for QY summary)
        """
        all_exported_files = {}

        # Export individual compounds (traces AND parameters)
        for compound_name, compound_data in compounds_data.items():
            replicate_results = compound_data.get('replicate_results', [])
            statistical_analyzer = compound_data.get('statistical_analyzer')

            if replicate_results:
                # Export replicate parameters (NEW - one file per compound with all replicates)
                param_file = self.export_individual_replicate_parameters(compound_name, replicate_results)
                if param_file:
                    if compound_name not in all_exported_files:
                        all_exported_files[compound_name] = {}
                    all_exported_files[compound_name]['replicate_parameters'] = param_file

                # Export replicate traces (OLD - individual trace files)
                if statistical_analyzer:
                    exported_files = self.export_compound(compound_name, replicate_results, statistical_analyzer)
                    if compound_name not in all_exported_files:
                        all_exported_files[compound_name] = {}
                    all_exported_files[compound_name].update(exported_files)

        # Export parameter statistics if available
        if analysis_results:
            param_stats_file = self.export_parameter_statistics(analysis_results)
            if param_stats_file:
                all_exported_files['parameter_statistics'] = param_stats_file

        # Export quantum yields if available
        if qy_results:
            qy_file = self.export_quantum_yields(qy_results)
            if qy_file:
                all_exported_files['quantum_yields'] = qy_file

        return all_exported_files
