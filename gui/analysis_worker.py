#!/usr/bin/env python3
"""
Analysis Worker - Background thread for running kinetics analysis pipeline

Pipeline:
1. Calculate SNR for all replicates
2. Filter replicates by SNR threshold
3. Run kinetics analysis (KineticsAnalyzer)
4. Calculate statistics (StatisticalAnalyzer)
5. Calculate quantum yields (quantum_yield_calculator)
"""

from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, List
import numpy as np

from core.snr_analyzer import SNRAnalyzer
from core.kinetics_analyzer import KineticsAnalyzer
from core.statistical_analyzer import StatisticalAnalyzer
from core.quantum_yield_calculator import calculate_quantum_yields_simple
from core.kinetics_dataclasses import KineticsResult
from utils.logger_config import get_logger

logger = get_logger(__name__)


class AnalysisWorker(QThread):
    """
    Background worker thread for complete analysis pipeline.

    Signals
    -------
    progress_update : pyqtSignal(int, str)
        Progress percentage and status message
    snr_calculated : pyqtSignal(str, int, float)
        Emitted when SNR is calculated for a replicate (compound, replicate_num, snr_linear)
    replicate_analyzed : pyqtSignal(str, int, dict)
        Emitted when a replicate is analyzed (compound, replicate_num, result_dict)
    analysis_complete : pyqtSignal(dict)
        Emitted when full analysis is complete with all results
    error_occurred : pyqtSignal(str)
        Emitted when error occurs
    """

    # Signals
    progress_update = pyqtSignal(int, str)
    snr_calculated = pyqtSignal(str, int, float)  # compound, replicate_num, snr_linear
    replicate_analyzed = pyqtSignal(str, int, dict)  # compound, replicate_num, result_dict
    analysis_complete = pyqtSignal(dict)  # full results
    error_occurred = pyqtSignal(str)

    def __init__(self, selected_replicates: Dict, snr_threshold: float, mode: str, mask_corrections: Dict = None):
        """
        Initialize analysis worker.

        Parameters
        ----------
        selected_replicates : dict
            From data_browser.get_selected_replicates()
            {compound_name: [replicate_data_dicts]}
        snr_threshold : float
            Minimum SNR (linear ratio) for analysis
        mode : str
            'homogeneous' or 'heterogeneous'
        mask_corrections : dict, optional
            User-specified mask corrections from preview plots
            {compound_name or compound_RepN: mask_end_time_us}
        """
        super().__init__()
        self.selected_replicates = selected_replicates
        self.snr_threshold = snr_threshold
        self.mode = mode
        self.mask_corrections = mask_corrections if mask_corrections is not None else {}
        self._is_running = True

    def run(self):
        """Execute analysis pipeline."""
        try:
            # Count total replicates
            total_replicates = sum(len(reps) for reps in self.selected_replicates.values())

            logger.info(f"Starting {self.mode} analysis: {len(self.selected_replicates)} compounds, "
                       f"{total_replicates} replicates, SNR threshold: {self.snr_threshold}:1")

            # Step 1: Calculate SNR for all replicates (25% of progress)
            self.progress_update.emit(0, "Calculating SNR...")
            snr_results = self._calculate_snr()

            if not self._is_running:
                return

            # Step 2: Filter by SNR threshold
            self.progress_update.emit(25, "Filtering by SNR threshold...")
            filtered_replicates, excluded_count = self._filter_by_snr(snr_results)

            logger.info(f"SNR filtering: {len(filtered_replicates)} passed, {excluded_count} excluded")

            if not filtered_replicates:
                self.error_occurred.emit(
                    f"No replicates meet SNR threshold of {self.snr_threshold}:1"
                )
                return

            if not self._is_running:
                return

            # Step 3: Run kinetics analysis (50% of progress)
            self.progress_update.emit(30, f"Analyzing kinetics ({len(filtered_replicates)} replicates)...")
            kinetics_results = self._run_kinetics_analysis(filtered_replicates)

            if not self._is_running:
                return

            # Step 4: Calculate statistics (80% of progress)
            self.progress_update.emit(80, "Calculating statistics...")
            statistics_results = self._calculate_statistics(kinetics_results)

            if not self._is_running:
                return

            # Step 5: Calculate quantum yields (95% of progress)
            self.progress_update.emit(95, "Calculating quantum yields...")
            qy_pairs, qy_results = self._calculate_quantum_yields(kinetics_results)

            # Complete
            self.progress_update.emit(100, "Analysis complete!")

            # Emit final results
            results = {
                'mode': self.mode,
                'snr_threshold': self.snr_threshold,
                'snr_results': snr_results,
                'kinetics_results': kinetics_results,
                'statistics_results': statistics_results,
                'qy_pairs': qy_pairs,
                'qy_results': qy_results,
                'excluded_count': excluded_count
            }

            self.analysis_complete.emit(results)
            logger.info("Analysis pipeline completed successfully")

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"Analysis failed:\n{error_detail}")
            self.error_occurred.emit(f"Analysis error: {str(e)}")

    def _calculate_snr(self) -> Dict:
        """
        Calculate SNR for all selected replicates.

        Returns
        -------
        dict
            {compound_name: [{replicate_data + 'snr_result': SNRResult}]}
        """
        snr_analyzer = SNRAnalyzer()
        snr_results = {}
        counter = 0
        total = sum(len(reps) for reps in self.selected_replicates.values())

        for compound_name, replicates in self.selected_replicates.items():
            snr_results[compound_name] = []

            for rep_data in replicates:
                if not self._is_running:
                    return snr_results

                counter += 1
                progress = int((counter / total) * 25)  # 0-25% for SNR calculation
                replicate_num = rep_data.get('replicate_index', 0) + 1

                # Calculate SNR
                time = rep_data['time']
                intensity = rep_data['intensity']
                decay_file = rep_data.get('decay_file')

                # Get dataset_type (user classification or 'auto')
                dataset_type = decay_file.dataset_type if decay_file and hasattr(decay_file, 'dataset_type') and decay_file.dataset_type else 'auto'

                snr_result = snr_analyzer.analyze_snr(time, intensity, dataset_type)

                # Emit SNR for data browser update
                self.snr_calculated.emit(compound_name, replicate_num, snr_result.snr_linear)

                # Store SNR result without copying entire replicate data (OPTIMIZED)
                # Instead of copying all arrays, just add snr_result to existing dict
                rep_data['snr_result'] = snr_result
                snr_results[compound_name].append(rep_data)

                self.progress_update.emit(
                    progress,
                    f"SNR: {compound_name} Rep {replicate_num} = {snr_result.snr_linear:.1f}:1"
                )

        return snr_results

    def _filter_by_snr(self, snr_results: Dict) -> tuple:
        """
        Filter replicates by SNR threshold.

        Returns
        -------
        tuple
            (filtered_dict, excluded_count)
        """
        filtered = {}
        excluded_count = 0

        for compound_name, replicates in snr_results.items():
            passed = []
            for rep_data in replicates:
                snr_linear = rep_data['snr_result'].snr_linear
                if snr_linear >= self.snr_threshold:
                    passed.append(rep_data)
                else:
                    excluded_count += 1
                    logger.warning(
                        f"{compound_name} Rep {rep_data.get('replicate_index', 0) + 1}: "
                        f"SNR {snr_linear:.1f}:1 below threshold {self.snr_threshold}:1 - EXCLUDED"
                    )

            if passed:
                filtered[compound_name] = passed

        return filtered, excluded_count

    def _run_kinetics_analysis(self, filtered_replicates: Dict) -> Dict:
        """
        Run kinetics analysis on filtered replicates.

        Returns
        -------
        dict
            {compound_name: [KineticsResult objects]}
        """
        analyzer = KineticsAnalyzer()
        kinetics_results = {}
        counter = 0
        total = sum(len(reps) for reps in filtered_replicates.values())

        for compound_name, replicates in filtered_replicates.items():
            kinetics_results[compound_name] = {
                'results': [],
                'wavelength': None,
                'classification': None
            }

            for rep_data in replicates:
                if not self._is_running:
                    return kinetics_results

                counter += 1
                progress = 30 + int((counter / total) * 50)  # 30-80% for kinetics
                replicate_num = rep_data.get('replicate_index', 0) + 1

                # Get data
                time = rep_data['time']
                intensity = rep_data['intensity']
                decay_file = rep_data['decay_file']

                # Get tau_delta
                tau_delta = decay_file.tau_delta_fixed if decay_file.tau_delta_fixed else 3.5

                # Get dataset_type (user classification or 'auto')
                dataset_type = decay_file.dataset_type if hasattr(decay_file, 'dataset_type') and decay_file.dataset_type else 'auto'

                # Check for custom mask correction
                # Keys: compound_name (applies to all replicates) or compound_RepN (specific replicate)
                specific_key = f"{compound_name}_Rep{replicate_num}"
                mask_correction = self.mask_corrections.get(specific_key) or self.mask_corrections.get(compound_name)

                # Run analysis with custom mask if available
                if mask_correction is not None:
                    logger.info(f"{compound_name} Rep{replicate_num}: Using custom mask correction {mask_correction:.4f} μs, dataset_type='{dataset_type}'")
                    result = analyzer.fit_kinetics(time, intensity, tau_delta_fixed=tau_delta, custom_mask_end_time=mask_correction, dataset_type=dataset_type)
                else:
                    logger.info(f"{compound_name} Rep{replicate_num}: dataset_type='{dataset_type}'")
                    result = analyzer.fit_kinetics(time, intensity, tau_delta_fixed=tau_delta, dataset_type=dataset_type)

                # Store result and metadata (from first replicate)
                kinetics_results[compound_name]['results'].append(result)
                if kinetics_results[compound_name]['wavelength'] is None:
                    kinetics_results[compound_name]['wavelength'] = decay_file.wavelength
                    kinetics_results[compound_name]['classification'] = decay_file.classification

                # Emit for live display
                result_dict = {
                    'compound': compound_name,
                    'replicate': replicate_num,
                    'result': result,
                    'classification': decay_file.classification,
                    'wavelength': decay_file.wavelength
                }
                self.replicate_analyzed.emit(compound_name, replicate_num, result_dict)

                self.progress_update.emit(
                    progress,
                    f"Analyzing: {compound_name} Rep {replicate_num}"
                )

        return kinetics_results

    def _calculate_statistics(self, kinetics_results: Dict) -> Dict:
        """
        Calculate statistics across replicates.

        Returns
        -------
        dict
            {compound_name: statistics_dict}
        """
        stat_analyzer = StatisticalAnalyzer()
        statistics_results = {}

        for compound_name, compound_data in kinetics_results.items():
            results_list = compound_data['results']
            if results_list:
                statistics = stat_analyzer.calculate_statistics(results_list)
                statistics_results[compound_name] = {
                    'statistics': statistics,
                    'n_replicates': len(results_list),
                    'wavelength': compound_data['wavelength'],
                    'classification': compound_data['classification'],
                    'mean_arrays': stat_analyzer.get_mean_arrays(),
                    'sd_arrays': stat_analyzer.get_sd_arrays()
                }

        return statistics_results

    def _calculate_quantum_yields(self, kinetics_results: Dict) -> tuple:
        """
        Calculate quantum yields.

        Returns
        -------
        tuple
            (qy_pairs, qy_results)
        """
        # Need to format kinetics_results for QY calculator
        # It expects: {compound: [compound_result_dict]} format
        formatted_results = {}

        for compound_name, compound_data in kinetics_results.items():
            results_list = compound_data['results']
            # Get first replicate's metadata (all share same compound info)
            if results_list:
                # We need to get the decay_file info - stored in selected_replicates
                # Find matching compound
                if compound_name in self.selected_replicates:
                    first_rep = self.selected_replicates[compound_name][0]
                    decay_file = first_rep['decay_file']

                    formatted_results[compound_name] = [{
                        'compound': compound_name,
                        'wavelength': compound_data['wavelength'],
                        'classification': compound_data['classification'],
                        'quantum_yield': decay_file.quantum_yield,
                        'quantum_yield_sd': decay_file.quantum_yield_sd,
                        'absorbance_at_wavelength': decay_file.absorbance_at_wavelength,
                        'replicate_results': results_list
                    }]

        # Calculate QY
        qy_pairs, qy_results = calculate_quantum_yields_simple(formatted_results)

        return qy_pairs, qy_results

    def stop(self):
        """Stop the analysis thread."""
        self._is_running = False
        logger.info("Analysis worker stop requested")
