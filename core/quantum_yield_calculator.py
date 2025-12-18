#!/usr/bin/env python3
"""
Simplified Quantum Yield Calculator for Singlet Oxygen Analysis
Uses only A-based method (amplitude from Main model fit)
REFACTORED: Works with KineticsResult dataclasses (v2.5 Phase 2)

QY_sample = QY_standard × (A_sample/A_standard) × (Abs_standard/Abs_sample)
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from utils.logger_config import get_logger
from core.kinetics_dataclasses import KineticsResult

logger = get_logger(__name__)


class QuantumYieldCalculator:
    """
    Simplified quantum yield calculator using only A-based amplitude method.

    QY formula:
        QY_sample = QY_std × (A_sample / A_std) × (Abs_std / Abs_sample)

    Where:
        - A = amplitude from Main model fit f(t-t0)
        - Abs = absorbance at excitation wavelength (solutions only, no scattering)
        - QY_std = literature quantum yield of standard compound
    """

    def __init__(self):
        """Initialize the quantum yield calculator."""
        pass

    def calculate_quantum_yields(self, analysis_results: Dict[str, List[Dict[str, Any]]]) \
            -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Calculate quantum yields for all valid sample-standard pairs.

        Args:
            analysis_results: Dictionary mapping compound names to list of analysis results
                Each result must contain:
                - wavelength: Excitation wavelength (nm)
                - classification: 'sample' or 'standard'
                - absorbance_at_wavelength: Absorbance value
                - quantum_yield: (standards only) Literature QY value
                - quantum_yield_sd: (standards only) Literature QY uncertainty
                - replicate_results: List of dicts with 'A' (amplitude) values

        Returns:
            (qy_pairs, qy_results): Lists of QY pair info and calculation results
        """
        # Find valid sample-standard pairs
        pairs = self._find_qy_pairs(analysis_results)

        if not pairs:
            logger.warning("No valid sample-standard pairs found")
            return [], []

        logger.info(f"Found {len(pairs)} sample-standard pairs for QY calculation")

        # Calculate QY for each pair
        qy_results = []
        for pair in pairs:
            result = self._calculate_single_qy(pair)
            qy_results.append(result)

        # Print summary
        successful = [r for r in qy_results if r.get('success', False)]
        failed = [r for r in qy_results if not r.get('success', False)]

        logger.info("Quantum Yield Calculation Summary:")
        logger.info(f"  Successful: {len(successful)}/{len(qy_results)}")
        logger.info(f"  Failed: {len(failed)}/{len(qy_results)}")

        if successful:
            for result in successful:
                sample = result['sample_compound']
                qy = result['quantum_yield']
                qy_err = result['quantum_yield_error']
                logger.info(f"  {sample}: QY = {qy:.4f} ± {qy_err:.4f}")

        return pairs, qy_results

    def _find_qy_pairs(self, analysis_results: Dict[str, List[Dict[str, Any]]]) \
            -> List[Dict[str, Any]]:
        """Find valid sample-standard pairs grouped by wavelength."""
        if not analysis_results:
            return []

        pairs = []
        wavelength_groups = self._group_by_wavelength(analysis_results)

        for wavelength, groups in wavelength_groups.items():
            samples = groups['samples']
            standards = groups['standards']

            if not samples or not standards:
                continue

            # Create all sample-standard pairs for this wavelength
            for sample in samples:
                for standard in standards:
                    if self._validate_pair(sample, standard):
                        pair = {
                            'wavelength': wavelength,
                            'sample': sample,
                            'standard': standard,
                            'sample_compound': sample.get('compound', 'Unknown'),
                            'standard_compound': standard.get('compound', 'Unknown'),
                            'standard_qy': standard.get('quantum_yield'),
                            'standard_qy_sd': standard.get('quantum_yield_sd', 0.0)
                        }
                        pairs.append(pair)

        return pairs

    def _group_by_wavelength(self, analysis_results: Dict[str, List[Dict[str, Any]]]) \
            -> Dict[float, Dict[str, List[Dict[str, Any]]]]:
        """Group analysis results by wavelength and classification."""
        wavelength_groups = {}

        for compound, results_list in analysis_results.items():
            for result in results_list:
                wavelength = result.get('wavelength')
                classification = result.get('classification')

                if wavelength is None or classification is None:
                    continue

                if wavelength not in wavelength_groups:
                    wavelength_groups[wavelength] = {'samples': [], 'standards': []}

                if classification.lower() == 'sample':
                    wavelength_groups[wavelength]['samples'].append(result)
                elif classification.lower() == 'standard':
                    wavelength_groups[wavelength]['standards'].append(result)

        return wavelength_groups

    def _validate_pair(self, sample: Dict[str, Any], standard: Dict[str, Any]) -> bool:
        """Validate that sample-standard pair has required data."""
        # Check absorbance values
        if not sample.get('absorbance_at_wavelength') or not standard.get('absorbance_at_wavelength'):
            return False

        # Check standard QY value
        if not standard.get('quantum_yield'):
            return False

        # Check for replicate results with A values
        sample_replicates = sample.get('replicate_results', [])
        standard_replicates = standard.get('replicate_results', [])

        if not sample_replicates or not standard_replicates:
            return False

        # Check that replicates have A values
        sample_A = []
        for rep in sample_replicates:
            A_val = rep.parameters.A if isinstance(rep, KineticsResult) else rep.get('A')
            if A_val and A_val > 0:
                sample_A.append(A_val)

        standard_A = []
        for rep in standard_replicates:
            A_val = rep.parameters.A if isinstance(rep, KineticsResult) else rep.get('A')
            if A_val and A_val > 0:
                standard_A.append(A_val)

        if not sample_A or not standard_A:
            return False

        return True

    def _calculate_single_qy(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quantum yield for a single sample-standard pair."""
        try:
            sample = pair['sample']
            standard = pair['standard']

            # Extract amplitudes (A values) from replicates
            sample_A = []
            for rep in sample['replicate_results']:
                A_val = rep.parameters.A if isinstance(rep, KineticsResult) else rep.get('A')
                if A_val and A_val > 0:
                    sample_A.append(float(A_val))

            standard_A = []
            for rep in standard['replicate_results']:
                A_val = rep.parameters.A if isinstance(rep, KineticsResult) else rep.get('A')
                if A_val and A_val > 0:
                    standard_A.append(float(A_val))

            # Extract absorbance values
            sample_abs_list = sample.get('absorbance_at_wavelength', [])
            standard_abs_list = standard.get('absorbance_at_wavelength', [])

            # Convert to list if single value
            if not isinstance(sample_abs_list, list):
                sample_abs_list = [sample_abs_list]
            if not isinstance(standard_abs_list, list):
                standard_abs_list = [standard_abs_list]

            # Standard QY parameters
            qy_standard = float(pair['standard_qy'])
            qy_standard_sd = float(pair.get('standard_qy_sd', 0.0))

            # Calculate QY for all replicate combinations
            qy_values = []
            abs_corrections = []

            for i, A_sample in enumerate(sample_A):
                for j, A_std in enumerate(standard_A):
                    # Get corresponding absorbance values
                    abs_sample = sample_abs_list[i] if i < len(sample_abs_list) else sample_abs_list[0]
                    abs_std = standard_abs_list[j] if j < len(standard_abs_list) else standard_abs_list[0]

                    # Calculate absorbance correction (Abs_std / Abs_sample)
                    abs_correction = abs_std / abs_sample if abs_sample > 0 else 1.0
                    abs_corrections.append(abs_correction)

                    # Calculate QY: QY_std × (A_sample/A_std) × (Abs_std/Abs_sample)
                    amplitude_ratio = A_sample / A_std
                    qy_replicate = qy_standard * amplitude_ratio * abs_correction
                    qy_values.append(qy_replicate)

            # Calculate statistics
            qy_mean = np.mean(qy_values)
            statistical_sd = np.std(qy_values, ddof=1) if len(qy_values) > 1 else 0.0

            # Systematic uncertainty from standard's literature uncertainty
            systematic_sd = qy_mean * (qy_standard_sd / qy_standard) if qy_standard > 0 and qy_standard_sd > 0 else 0.0

            # Combined uncertainty (propagated)
            qy_sd = np.sqrt(statistical_sd**2 + systematic_sd**2)
            qy_rel_error = (qy_sd / qy_mean * 100) if qy_mean > 0 else 0.0

            return {
                'success': True,
                'sample_compound': pair['sample_compound'],
                'standard_compound': pair['standard_compound'],
                'wavelength': pair['wavelength'],
                'quantum_yield': qy_mean,
                'quantum_yield_error': qy_sd,
                'statistical_error': statistical_sd,
                'systematic_error': systematic_sd,
                'relative_error_percent': qy_rel_error,
                'n_calculations': len(qy_values),
                'n_sample_replicates': len(sample_A),
                'n_standard_replicates': len(standard_A),
                'absorption_correction': np.mean(abs_corrections),
                'method': 'A_based'
            }

        except Exception as e:
            return {
                'success': False,
                'sample_compound': pair.get('sample_compound', 'Unknown'),
                'standard_compound': pair.get('standard_compound', 'Unknown'),
                'wavelength': pair.get('wavelength'),
                'error': str(e)
            }


# Backward compatibility function
def calculate_quantum_yields_simple(analysis_results: Dict[str, List[Dict[str, Any]]]) \
        -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Legacy function name for backward compatibility.

    Calculates quantum yields using A-based method only.
    """
    calculator = QuantumYieldCalculator()
    return calculator.calculate_quantum_yields(analysis_results)
