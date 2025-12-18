#!/usr/bin/env python3
"""
File Parser for Singlet Oxygen Kinetics Analyzer
Handles parsing of experimental data files with comprehensive validation and linking.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict
from utils.logger_config import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedFile:
    """Container for parsed file information."""
    compound: str
    file_type: str  # 'decay' or 'absorption'
    file_path: str

    # Decay-specific fields
    wavelength: Optional[float] = None
    tau_delta_fixed: Optional[float] = None
    quantum_yield: Optional[float] = None
    quantum_yield_sd: Optional[float] = None
    excitation_intensity: Optional[float] = None  # EI value (power/energy)
    intensity_unit: Optional[str] = None  # Unit (e.g., 'mW', 'mJ', 'uW', 'W')
    classification: Optional[str] = None  # 'Standard' or 'Sample'

    # Absorption-specific fields
    absorbance_at_wavelength: Optional[Union[float, List[float]]] = None

    # Dataset classification (user-specified or auto-detected)
    dataset_type: Optional[str] = None  # 'lag_spike', 'spike_only', 'clean_signal', 'preprocessed', or None (ask user)

    # Data
    data: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """Set classification based on quantum yield presence."""
        pass
    
    def get_kinetics_data(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Convert data format ensuring all replicates align to same time points.

        NaN values in intensity columns are preserved (user may have removed spike/lag manually).
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available")

        time_data = self.data.iloc[:, 0].values

        # NO NaN removal - keep data as-is
        # If user removed spike/lag manually, they'll have NaN in intensity
        # kinetics_analyzer will detect and handle this appropriately

        intensity_replicates = []
        for col_idx in range(1, self.data.shape[1]):
            intensity_data = self.data.iloc[:, col_idx].values
            intensity_replicates.append(intensity_data)

        return time_data, intensity_replicates

    def get_absorbance_for_replicate(self, replicate_index: int) -> Optional[float]:
        """Get absorbance value for specific replicate."""
        if self.absorbance_at_wavelength is None:
            return None
            
        # Single value for all replicates
        if isinstance(self.absorbance_at_wavelength, (int, float)):
            return float(self.absorbance_at_wavelength)
            
        # List of values per replicate
        if isinstance(self.absorbance_at_wavelength, list):
            if 0 <= replicate_index < len(self.absorbance_at_wavelength):
                return self.absorbance_at_wavelength[replicate_index]
            else:
                logger.warning(f"Replicate index {replicate_index} out of range for absorbance data")
                return None
                
        return None


class FileParseError(Exception):
    """Raised when file parsing fails."""
    pass


class FileParser:
    """Optimized file parser with caching and improved performance."""
    
    def __init__(self):
        """Initialize parser with patterns supporting dots in names."""
        # Updated patterns to handle dots in compound names and parameters
        # Format: Decay_[Compound]_EX[λ]nm_tauD[value]_QY[value]_QYsd[sd]_EI[value][unit].csv
        self.decay_pattern = (
            r'Decay_(.+?)_EX(\d+(?:\.\d+)?)nm'
            r'(?:_tauD(\d+(?:\.\d+)?))?'
            r'(?:_QY(\d+(?:\.\d+)?))?'
            r'(?:_QYsd(\d+(?:\.\d+)?))?'
            r'(?:_EI(\d+(?:\.\d+)?)([a-zA-Z]+))?'  # Optional: EI value + unit
            r'\.csv'
        )

        self.abs_pattern = r'Abs_(.+?)\.csv'

        # Cache for CSV parsing attempts with size limit (optimized in Session 41)
        # For 8 GB RAM systems, limit to 50 files (~25-50 MB typical)
        self.MAX_CACHE_SIZE = 50
        self._csv_cache = OrderedDict()  # LRU cache: OrderedDict + move_to_end()
        self._delimiter_cache = {}

    def _add_to_cache(self, file_path: str, dataframe: pd.DataFrame):
        """Add item to cache with LRU eviction if full."""
        if len(self._csv_cache) >= self.MAX_CACHE_SIZE:
            # Remove oldest item (FIFO/LRU behavior)
            oldest_key = next(iter(self._csv_cache))
            removed = self._csv_cache.pop(oldest_key)
            logger.debug(f"Cache full, evicted: {Path(oldest_key).name} ({len(removed)} rows)")

        self._csv_cache[file_path] = dataframe
        logger.debug(f"Cached: {Path(file_path).name} ({len(dataframe)} rows), cache size: {len(self._csv_cache)}/{self.MAX_CACHE_SIZE}")

    def clear_cache(self):
        """Clear all cached data to free memory and prevent stale data."""
        cache_size = len(self._csv_cache)
        memory_estimate = sum(df.memory_usage(deep=True).sum() for df in self._csv_cache.values()) / (1024**2)
        self._csv_cache.clear()
        self._delimiter_cache.clear()
        # Clear lru_cache for _get_optimal_delimiter
        if hasattr(self._get_optimal_delimiter, 'cache_clear'):
            self._get_optimal_delimiter.cache_clear()
        logger.info(f"Cleared file parser cache ({cache_size} entries, ~{memory_estimate:.1f} MB)")

    def parse_directory(self, directory: str) -> Dict[str, List[ParsedFile]]:
        """Parse all CSV files in directory and organize by compound."""
        # Clear cache before parsing new directory to prevent stale data
        self.clear_cache()

        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileParseError(f"Directory does not exist: {directory}")
            
        compounds = {}
        csv_files = list(directory_path.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {directory}")
            return compounds

        logger.info(f"Found {len(csv_files)} CSV files in {directory}")
        
        for file_path in csv_files:
            try:
                parsed_file = self.parse_file(str(file_path))
                compound = parsed_file.compound
                
                if compound not in compounds:
                    compounds[compound] = []
                    
                compounds[compound].append(parsed_file)
                
            except FileParseError as e:
                logger.warning(f"Could not parse {file_path.name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error parsing {file_path.name}: {e}")
                continue

        logger.info(f"Successfully parsed {len(compounds)} compounds")
        return compounds
        
    def parse_file(self, file_path: str) -> ParsedFile:
        """Parse single file and extract all information."""
        file_path = Path(file_path)
        filename = file_path.name
        
        # Determine file type and parse filename
        if 'decay' in filename.lower():
            parsed_info = self._parse_decay_filename(filename)
        elif 'abs' in filename.lower():
            parsed_info = self._parse_abs_filename(filename)
        else:
            raise FileParseError(f"Cannot determine file type from filename: {filename}")
            
        # Load data with caching
        try:
            data = self._load_csv_data_cached(str(file_path))
        except Exception as e:
            raise FileParseError(f"Failed to load data from {filename}: {e}")
            
        # Create ParsedFile object
        parsed_file = ParsedFile(
            compound=parsed_info['compound'],
            file_type=parsed_info['file_type'],
            file_path=str(file_path),
            data=data
        )
        
        # Add type-specific information
        if parsed_file.file_type == 'decay':
            parsed_file.wavelength = parsed_info['wavelength']
            parsed_file.tau_delta_fixed = parsed_info['tau_delta_fixed']
            parsed_file.quantum_yield = parsed_info['quantum_yield']
            parsed_file.quantum_yield_sd = parsed_info['quantum_yield_sd']
            parsed_file.excitation_intensity = parsed_info['excitation_intensity']
            parsed_file.intensity_unit = parsed_info['intensity_unit']
            # Set classification after all fields are populated
            parsed_file.classification = 'Standard' if parsed_file.quantum_yield is not None else 'Sample'

        return parsed_file
        
    def _parse_decay_filename(self, filename: str) -> Dict[str, Union[str, float]]:
        """Parse decay filename supporting dots in compound names and EI parameter."""
        match = re.search(self.decay_pattern, filename, re.IGNORECASE)

        if not match:
            raise FileParseError(f"Decay filename '{filename}' doesn't match expected format")

        compound = match.group(1)
        wavelength = float(match.group(2))
        tau_delta_str = match.group(3)
        qy_str = match.group(4)
        qy_sd_str = match.group(5)
        ei_value_str = match.group(6)  # EI value
        ei_unit_str = match.group(7)   # EI unit

        # Parse optional parameters
        tau_delta_fixed = float(tau_delta_str) if tau_delta_str else None
        quantum_yield = float(qy_str) if qy_str else None
        quantum_yield_sd = float(qy_sd_str) if qy_sd_str else None
        excitation_intensity = float(ei_value_str) if ei_value_str else None
        intensity_unit = ei_unit_str if ei_unit_str else None

        # Log EI parsing for debugging
        logger.debug(f"Parsed {filename}: EI={excitation_intensity}, unit={intensity_unit}")

        return {
            'compound': compound,
            'wavelength': wavelength,
            'tau_delta_fixed': tau_delta_fixed,
            'quantum_yield': quantum_yield,
            'quantum_yield_sd': quantum_yield_sd,
            'excitation_intensity': excitation_intensity,
            'intensity_unit': intensity_unit,
            'file_type': 'decay'
        }
        
    def _parse_abs_filename(self, filename: str) -> Dict[str, str]:
        """Parse absorption filename supporting dots in compound names."""
        match = re.search(self.abs_pattern, filename, re.IGNORECASE)
        
        if not match:
            raise FileParseError(f"Absorption filename '{filename}' doesn't match expected format")
            
        return {
            'compound': match.group(1),
            'file_type': 'absorption'
        }
        
    @lru_cache(maxsize=128)
    def _get_optimal_delimiter(self, file_path: str) -> str:
        """Determine optimal CSV delimiter with caching."""
        delimiters = [',', '\t', ';']
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample_lines = [f.readline() for _ in range(3)]
        
        sample_text = ''.join(sample_lines)
        
        # Count occurrences of each delimiter
        delimiter_scores = {}
        for delimiter in delimiters:
            count = sample_text.count(delimiter)
            if count > 0:
                # Check if delimiter gives consistent column count
                line_counts = [line.count(delimiter) for line in sample_lines if line.strip()]
                if line_counts and len(set(line_counts)) == 1:  # Consistent column count
                    delimiter_scores[delimiter] = count
        
        if delimiter_scores:
            return max(delimiter_scores, key=delimiter_scores.get)
        
        return ','  # Default fallback
        
    def _load_csv_data_cached(self, file_path: str) -> pd.DataFrame:
        """Load CSV data with caching and optimized delimiter detection."""
        if file_path in self._csv_cache:
            # Move to end for LRU (most recently used stays at end)
            self._csv_cache.move_to_end(file_path)
            return self._csv_cache[file_path]

        # Get optimal delimiter
        delimiter = self._get_optimal_delimiter(file_path)

        try:
            data = pd.read_csv(file_path, header=None, delimiter=delimiter)

            # Validate data dimensions
            if data.shape[1] < 2 or data.shape[0] < 10:
                raise ValueError(f"Insufficient data: {data.shape}")

            # Convert to numeric
            numeric_data = data.apply(pd.to_numeric, errors='coerce')
            numeric_data = numeric_data.dropna(how='all')

            if numeric_data.shape[0] < 10:
                raise ValueError("Insufficient numeric data after cleanup")

            # Cache the result using LRU cache with size limit
            self._add_to_cache(file_path, numeric_data)
            return numeric_data
            
        except Exception as e:
            # Fallback: try all delimiters
            return self._load_csv_fallback(file_path)
        
    def _load_csv_fallback(self, file_path: str) -> pd.DataFrame:
        """Fallback CSV loading with multiple delimiter attempts."""
        delimiters = [',', '\t', ';']
        
        for delimiter in delimiters:
            try:
                data = pd.read_csv(file_path, header=None, delimiter=delimiter)
                
                if data.shape[1] >= 2 and data.shape[0] >= 10:
                    numeric_data = data.apply(pd.to_numeric, errors='coerce')
                    numeric_data = numeric_data.dropna(how='all')

                    if numeric_data.shape[0] >= 10:
                        self._add_to_cache(file_path, numeric_data.copy())
                        return numeric_data
                        
            except Exception:
                continue
                
        raise FileParseError(f"Could not parse CSV data from {file_path}")
        
    def link_absorption_data(self, parsed_files: List[ParsedFile], directory: str) -> None:
        """Link decay files with absorption data, supporting multi-column matching."""
        if not parsed_files:
            return
            
        compound = parsed_files[0].compound
        abs_file_path = self._find_absorption_file(compound, directory)
        
        if abs_file_path is None:
            logger.warning(f"No absorption file found for compound '{compound}'")
            return
            
        try:
            abs_parsed_file = self.parse_file(abs_file_path)
            abs_data = abs_parsed_file.data
            
            for parsed_file in parsed_files:
                if parsed_file.file_type == 'decay' and parsed_file.wavelength is not None:
                    try:
                        # Get number of replicates in decay file
                        n_decay_replicates = parsed_file.data.shape[1] - 1
                        
                        # Extract absorbance with proper column matching
                        absorbance = self._extract_absorbance_optimized(
                            abs_data, parsed_file.wavelength, n_decay_replicates
                        )
                        parsed_file.absorbance_at_wavelength = absorbance
                        
                        # Log linking information
                        self._log_absorbance_linking(compound, absorbance, parsed_file.wavelength, n_decay_replicates)
                        
                    except FileParseError as e:
                        logger.warning(f"Could not extract absorbance for {parsed_file.file_path}: {e}")

        except FileParseError as e:
            logger.warning(f"Could not load absorption file {abs_file_path}: {e}")
            
    def _find_absorption_file(self, compound: str, directory: str) -> Optional[str]:
        """Find absorption file for given compound."""
        directory_path = Path(directory)
        
        target_filename = f"Abs_{compound}.csv"
        target_path = directory_path / target_filename
        
        if target_path.exists():
            return str(target_path)
            
        # Case-insensitive search
        for file_path in directory_path.glob("*.csv"):
            if file_path.name.lower() == target_filename.lower():
                return str(file_path)
                
        return None
        
    def _extract_absorbance_optimized(self, abs_data: pd.DataFrame, 
                                    target_wavelength: float, 
                                    n_decay_replicates: int,
                                    tolerance: float = 2.0) -> Union[float, List[float]]:
        """Extract absorbance values with optimized wavelength matching."""
        if abs_data.shape[1] < 2:
            raise FileParseError("Absorption data must have at least 2 columns (wavelength, absorbance)")
            
        wavelengths = abs_data.iloc[:, 0]
        n_abs_columns = abs_data.shape[1] - 1
        
        # Find closest wavelength match
        wavelength_diff = np.abs(wavelengths - target_wavelength)
        min_diff_idx = wavelength_diff.idxmin()
        min_diff = wavelength_diff.iloc[min_diff_idx]
        
        if min_diff > tolerance:
            raise FileParseError(
                f"No absorbance data within {tolerance} nm of {target_wavelength} nm. "
                f"Closest is {wavelengths.iloc[min_diff_idx]:.1f} nm ({min_diff:.1f} nm away)"
            )
        
        # Extract absorbance values based on column structure
        if n_abs_columns == 1:
            # Single absorbance column for all replicates
            absorbance = abs_data.iloc[min_diff_idx, 1]
            if pd.isna(absorbance):
                raise FileParseError(f"Invalid absorbance value at {wavelengths.iloc[min_diff_idx]:.1f} nm")
            return float(absorbance)
        
        elif n_abs_columns == n_decay_replicates:
            # One absorbance column per decay replicate
            absorbances = []
            for col_idx in range(1, abs_data.shape[1]):
                absorbance = abs_data.iloc[min_diff_idx, col_idx]
                if pd.isna(absorbance):
                    raise FileParseError(f"Invalid absorbance value in column {col_idx} at {wavelengths.iloc[min_diff_idx]:.1f} nm")
                absorbances.append(float(absorbance))
            return absorbances
        
        elif n_abs_columns > n_decay_replicates:
            # More absorbance columns than decay replicates - use first n_decay_replicates
            logger.warning(f"Absorption file has {n_abs_columns} columns, using first {n_decay_replicates} for {n_decay_replicates} decay replicates")
            absorbances = []
            for col_idx in range(1, min(n_decay_replicates + 1, abs_data.shape[1])):
                absorbance = abs_data.iloc[min_diff_idx, col_idx]
                if pd.isna(absorbance):
                    raise FileParseError(f"Invalid absorbance value in column {col_idx}")
                absorbances.append(float(absorbance))
            return absorbances
        
        else:
            # Fewer absorbance columns than decay replicates - use first column for all
            logger.warning(f"Absorption file has {n_abs_columns} columns but decay has {n_decay_replicates} replicates. Using first absorbance column for all replicates.")
            absorbance = abs_data.iloc[min_diff_idx, 1]
            if pd.isna(absorbance):
                raise FileParseError(f"Invalid absorbance value at {wavelengths.iloc[min_diff_idx]:.1f} nm")
            return float(absorbance)
        
    def _log_absorbance_linking(self, compound: str, absorbance: Union[float, List[float]],
                              wavelength: float, n_decay_replicates: int) -> None:
        """Log absorbance linking information."""
        if isinstance(absorbance, list):
            logger.info(f"Linked {compound}: {len(absorbance)} replicate-specific absorbances at {wavelength} nm")
            if len(absorbance) <= 10:  # Only show details for reasonable number of replicates
                abs_pairs = [f'Rep{i+1}→{abs:.3f}' for i, abs in enumerate(absorbance)]
                logger.info(f"  Values: {', '.join(abs_pairs)}")
        else:
            logger.info(f"Linked {compound}: Single absorbance = {absorbance:.3f} at {wavelength} nm (for all {n_decay_replicates} replicates)")
        
    def validate_file_structure(self, directory: str) -> Dict[str, Any]:
        """Validate file structure and provide comprehensive report."""
        try:
            compounds = self.parse_directory(directory)
            
            validation = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'compounds': len(compounds),
                'summary': {},
                'recommendations': []
            }
            
            if len(compounds) == 0:
                validation['valid'] = False
                validation['errors'].append("No valid compound data found")
                validation['recommendations'].append("Check filename format: Decay_[Compound]_EX[λ]nm_...")
                return validation
                
            for compound, files in compounds.items():
                self.link_absorption_data(files, directory)
                validation['summary'][compound] = self._validate_compound_files(compound, files, validation)
                
            self._generate_validation_recommendations(validation)
                
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Directory validation failed: {e}"],
                'warnings': [],
                'compounds': 0,
                'summary': {},
                'recommendations': ['Check directory path and file permissions']
            }
    
    def _validate_compound_files(self, compound: str, files: List[ParsedFile], validation: Dict) -> Dict:
        """Validate files for single compound."""
        decay_files = [f for f in files if f.file_type == 'decay']
        abs_files = [f for f in files if f.file_type == 'absorption']
        standards = [f for f in decay_files if f.classification == 'Standard']
        samples = [f for f in decay_files if f.classification == 'Sample']
        
        compound_summary = {
            'total_files': len(files),
            'decay_files': len(decay_files),
            'absorption_files': len(abs_files),
            'standards': len(standards),
            'samples': len(samples),
            'has_absorption_data': any(f.absorbance_at_wavelength is not None for f in decay_files),
            'qy_calculation_possible': len(standards) > 0 and len(samples) > 0
        }
        
        # Check for issues
        if len(decay_files) == 0:
            validation['errors'].append(f"No valid decay files found for {compound}")
            validation['valid'] = False
            
        if len(abs_files) == 0:
            validation['warnings'].append(f"No absorption file found for {compound}")
            
        # Validate data quality
        for decay_file in decay_files:
            if decay_file.data is not None:
                n_points = len(decay_file.data)
                n_replicates = decay_file.data.shape[1] - 1
                
                if n_points < 50:
                    validation['warnings'].append(f"{compound}: Only {n_points} data points (recommend >50)")
                    
                if n_replicates < 3:
                    validation['warnings'].append(f"{compound}: Only {n_replicates} replicates (recommend 3-9)")
                elif n_replicates > 9:
                    validation['warnings'].append(f"{compound}: {n_replicates} replicates (may be excessive)")
        
        return compound_summary
    
    def _generate_validation_recommendations(self, validation: Dict) -> None:
        """Generate recommendations based on validation results."""
        if validation['valid']:
            total_standards = sum(s['standards'] for s in validation['summary'].values())
            total_samples = sum(s['samples'] for s in validation['summary'].values())
            
            if total_standards > 0 and total_samples > 0:
                validation['recommendations'].append("Ready for quantum yield calculations")
            elif total_standards == 0:
                validation['recommendations'].append("Add standards with known QY values for quantum yield calculations")
            elif total_samples == 0:
                validation['recommendations'].append("Add samples for quantum yield calculations")
                
        if not validation['recommendations']:
            validation['recommendations'].append("File structure validation complete")


def validate_directory_structure(directory: str) -> None:
    """Convenience function to validate and print directory structure."""
    parser = FileParser()
    validation = parser.validate_file_structure(directory)
    
    print(f"\nDIRECTORY VALIDATION REPORT")
    print(f"Directory: {directory}")
    print(f"Status: {'VALID' if validation['valid'] else 'INVALID'}")
    print(f"Compounds found: {validation['compounds']}")
    
    if validation['errors']:
        print(f"\nERRORS:")
        for error in validation['errors']:
            print(f"  {error}")
            
    if validation['warnings']:
        print(f"\nWARNINGS:")
        for warning in validation['warnings']:
            print(f"  {warning}")
            
    if validation['recommendations']:
        print(f"\nRECOMMENDATIONS:")
        for rec in validation['recommendations']:
            print(f"  {rec}")
            
    if validation['summary']:
        print(f"\nCOMPOUND SUMMARY:")
        for compound, info in validation['summary'].items():
            print(f"  {compound}:")
            print(f"    Decay files: {info['decay_files']}")
            print(f"    Standards: {info['standards']}, Samples: {info['samples']}")
            print(f"    Absorption data: {'Yes' if info['has_absorption_data'] else 'No'}")
            print(f"    QY calculation: {'Yes' if info.get('qy_calculation_possible', False) else 'No'}")