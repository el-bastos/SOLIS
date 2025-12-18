#!/usr/bin/env python3
"""
Session Save/Load Manager for SOLIS
====================================

Provides functionality to save and restore complete SOLIS analysis sessions.
Sessions are stored as JSON files containing:
- Loaded data file paths
- Analysis results (homogeneous, heterogeneous, surplus, vesicle)
- User preferences and settings
- Plot configurations
- Mask corrections

This enables:
- Session continuity across application restarts
- Reproducible analysis workflows
- Sharing results with collaborators
- Quick resume after interruptions

Author: SOLIS Team
Date: 2025-10-31
"""

import json
import numpy as np
import pickle
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import asdict, is_dataclass, fields

from core.kinetics_dataclasses import (
    KineticsResult, FitParameters, SNRResult, FitQuality,
    HeterogeneousFitResult, LiteratureModelResult, WorkflowInfo
)
from heterogeneous.heterogeneous_dataclasses import (
    VesicleGeometry, DiffusionParameters, SimulationResult,
    HeterogeneousFitResult as HeteroFitResultNew  # New heterogeneous dataclass
)
from surplus.surplus_analyzer import SurplusResult
from data.file_parser import ParsedFile
from utils.logger_config import get_logger

logger = get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy arrays, pandas DataFrames, and dataclasses."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__type__': 'ndarray',
                'data': obj.tolist(),
                'dtype': str(obj.dtype),
                'shape': obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif is_dataclass(obj):
            # Use custom serialization to preserve nested dataclass type markers
            class_name = obj.__class__.__name__
            logger.debug(f"Serializing dataclass: {class_name}")
            return {
                '__type__': 'dataclass',
                'class': class_name,
                'data': self._serialize_dataclass_fields(obj)
            }
        elif isinstance(obj, Path):
            return {
                '__type__': 'Path',
                'path': str(obj)
            }
        # Handle pandas DataFrame using pickle for dtype preservation
        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                # Serialize using pickle (preserves dtypes, handles integer column names)
                pickled = pickle.dumps(obj)
                encoded = base64.b64encode(pickled).decode('utf-8')
                return {
                    '__type__': 'DataFrame',
                    'pickled_data': encoded,
                    'shape': obj.shape,
                    'columns': list(obj.columns)  # For verification/debugging
                }
        except ImportError:
            pass
        return super().default(obj)

    def _serialize_dataclass_fields(self, obj):
        """
        Serialize dataclass fields while preserving nested dataclass type information.

        This replaces asdict() to ensure nested dataclasses retain their '__type__'
        markers, which are needed for proper reconstruction during deserialization.

        Args:
            obj: Dataclass instance to serialize

        Returns:
            Dictionary with serialized fields
        """
        result = {}
        for field_obj in fields(obj):
            value = getattr(obj, field_obj.name)

            if value is None:
                # Handle None values
                result[field_obj.name] = None
            elif is_dataclass(value):
                # Recursively serialize nested dataclass (preserves __type__ marker)
                result[field_obj.name] = self.default(value)
            elif isinstance(value, np.ndarray):
                # Serialize numpy arrays
                result[field_obj.name] = self.default(value)
            elif isinstance(value, dict):
                # Recursively handle dicts that might contain dataclasses or arrays
                result[field_obj.name] = self._serialize_dict_values(value)
            elif isinstance(value, (list, tuple)):
                # Recursively handle lists/tuples that might contain dataclasses or arrays
                result[field_obj.name] = self._serialize_sequence(value)
            elif isinstance(value, Path):
                # Handle Path objects
                result[field_obj.name] = self.default(value)
            else:
                # Primitive types, strings, numbers, etc.
                result[field_obj.name] = value

        return result

    def _serialize_dict_values(self, d: dict) -> dict:
        """Recursively serialize dictionary values."""
        result = {}
        for key, value in d.items():
            if is_dataclass(value):
                result[key] = self.default(value)
            elif isinstance(value, np.ndarray):
                result[key] = self.default(value)
            elif isinstance(value, dict):
                result[key] = self._serialize_dict_values(value)
            elif isinstance(value, (list, tuple)):
                result[key] = self._serialize_sequence(value)
            else:
                result[key] = value
        return result

    def _serialize_sequence(self, seq) -> list:
        """Recursively serialize list/tuple elements."""
        result = []
        for item in seq:
            if is_dataclass(item):
                result.append(self.default(item))
            elif isinstance(item, np.ndarray):
                result.append(self.default(item))
            elif isinstance(item, dict):
                result.append(self._serialize_dict_values(item))
            elif isinstance(item, (list, tuple)):
                result.append(self._serialize_sequence(item))
            else:
                result.append(item)
        return result


def numpy_decoder(dct):
    """
    Custom JSON decoder that reconstructs NumPy arrays, pandas DataFrames, and Paths.

    Note: Dataclass reconstruction is handled separately in post-processing.
    """
    if '__type__' in dct:
        if dct['__type__'] == 'ndarray':
            return np.array(dct['data'], dtype=dct['dtype']).reshape(dct['shape'])
        elif dct['__type__'] == 'Path':
            return Path(dct['path'])
        elif dct['__type__'] == 'DataFrame':
            # Reconstruct pandas DataFrame from pickle (preserves dtypes)
            try:
                import pandas as pd
                # Check if using new pickle format (has 'pickled_data')
                if 'pickled_data' in dct:
                    decoded = base64.b64decode(dct['pickled_data'].encode('utf-8'))
                    df = pickle.loads(decoded)
                    return df
                else:
                    # Legacy format (JSON-based) - will have dtype issues
                    logger.warning("Loading DataFrame from legacy JSON format - dtypes may be corrupted")
                    return pd.DataFrame(dct['data'], columns=dct['columns'], index=dct['index'])
            except ImportError:
                # If pandas not available, return as dict
                logger.error("Pandas not available - cannot deserialize DataFrame")
                return dct.get('data', {})
        # Note: dataclass reconstruction moved to post-processing
    return dct


class SessionManager:
    """
    Manages saving and loading SOLIS analysis sessions.

    Session file format (.solis.json):
    {
        "metadata": {
            "version": "1.0",
            "created": "2025-10-31T10:30:00",
            "solis_version": "1.0.0",
            "description": "Optional user description"
        },
        "data": {
            "folder_path": "path/to/data",
            "loaded_files": [...],
            "compounds": {...}
        },
        "analysis": {
            "homogeneous": {...},
            "heterogeneous": {...},
            "surplus": {...},
            "vesicle": {...}
        },
        "preferences": {
            "snr_thresholds": {...},
            "surplus": {...}
        },
        "ui_state": {
            "mask_corrections": {...},
            "selected_compounds": [...]
        }
    }
    """

    VERSION = "1.0"

    @staticmethod
    def save_session(
        filepath: Path,
        loaded_compounds: Optional[Dict] = None,
        analysis_results: Optional[Dict] = None,
        heterogeneous_results: Optional[Dict] = None,
        surplus_results: Optional[Dict] = None,
        preferences: Optional[Dict] = None,
        mask_corrections: Optional[Dict] = None,
        folder_path: Optional[Path] = None,
        description: str = "",
        plot_windows: Optional[List[Dict]] = None,
        plot_operations: Optional[List[Dict]] = None
    ) -> bool:
        """
        Save current SOLIS session to JSON file.

        Args:
            filepath: Path to save session file (.solis.json)
            loaded_compounds: Dictionary of loaded compound data
            analysis_results: Dictionary of homogeneous analysis results
            heterogeneous_results: Dictionary of heterogeneous/vesicle analysis results
            surplus_results: Dictionary of surplus analysis results
            preferences: User preferences dictionary
            mask_corrections: Mask correction dictionary
            folder_path: Path to data folder
            description: Optional session description
            plot_windows: List of plot window states (positions, sizes, zoom)
            plot_operations: List of plot operations for replay on load

        Returns:
            True if save successful, False otherwise
        """
        try:
            logger.info(f"Saving session to {filepath}")

            # Build session data structure
            session_data = {
                "metadata": {
                    "version": SessionManager.VERSION,
                    "created": datetime.now().isoformat(),
                    "solis_version": "1.0.0",  # TODO: Get from version file
                    "description": description
                },
                "data": {
                    "folder_path": str(folder_path) if folder_path else None,
                    "loaded_compounds": SessionManager._serialize_compounds(loaded_compounds),
                },
                "analysis": {
                    "homogeneous": SessionManager._serialize_analysis(analysis_results),
                    "heterogeneous": SessionManager._serialize_analysis(heterogeneous_results),
                    "surplus": SessionManager._serialize_analysis(surplus_results),
                },
                "preferences": preferences or {},
                "ui_state": {
                    "mask_corrections": mask_corrections or {},
                    "plot_windows": plot_windows or [],
                    "plot_operations": plot_operations or []
                }
            }

            # Write to file with pretty formatting
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, cls=NumpyEncoder, indent=2)

            logger.info(f"Session saved successfully: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}", exc_info=True)
            return False

    @staticmethod
    def load_session(filepath: Path) -> Optional[Dict[str, Any]]:
        """
        Load SOLIS session from JSON file.

        Args:
            filepath: Path to session file

        Returns:
            Dictionary containing session data, or None if load failed
            Structure:
            {
                "metadata": {...},
                "data": {...},
                "analysis": {...},
                "preferences": {...},
                "ui_state": {...}
            }
        """
        try:
            logger.info(f"Loading session from {filepath}")

            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f, object_hook=numpy_decoder)

            # Validate version
            version = session_data.get('metadata', {}).get('version')
            if version != SessionManager.VERSION:
                logger.warning(f"Session version mismatch: {version} vs {SessionManager.VERSION}")
                # For now, proceed anyway - add migration logic here if needed

            # Deserialize loaded compounds back to ParsedFile format
            if 'data' in session_data and 'loaded_compounds' in session_data['data']:
                serialized_compounds = session_data['data']['loaded_compounds']
                session_data['data']['loaded_compounds'] = SessionManager._deserialize_compounds(serialized_compounds)

            # Post-process analysis results to reconstruct dataclasses
            if 'analysis' in session_data:
                logger.info("Post-processing analysis results to reconstruct dataclasses...")

                # Reconstruct homogeneous results
                if 'homogeneous' in session_data['analysis']:
                    session_data['analysis']['homogeneous'] = SessionManager._reconstruct_analysis_results(
                        session_data['analysis']['homogeneous']
                    )

                # Reconstruct heterogeneous results
                if 'heterogeneous' in session_data['analysis']:
                    session_data['analysis']['heterogeneous'] = SessionManager._reconstruct_analysis_results(
                        session_data['analysis']['heterogeneous']
                    )

                # Reconstruct surplus results
                if 'surplus' in session_data['analysis']:
                    session_data['analysis']['surplus'] = SessionManager._reconstruct_analysis_results(
                        session_data['analysis']['surplus']
                    )

            logger.info(f"Session loaded successfully from {filepath}")
            logger.info(f"Session created: {session_data['metadata'].get('created')}")
            if session_data['metadata'].get('description'):
                logger.info(f"Description: {session_data['metadata']['description']}")

            return session_data

        except Exception as e:
            logger.error(f"Failed to load session: {e}", exc_info=True)
            return None

    @staticmethod
    def _serialize_compounds(compounds: Optional[Dict]) -> Optional[Dict]:
        """
        Serialize compound data for JSON storage.

        Converts NumPy arrays and dataclasses (ParsedFile objects) to JSON-compatible format.

        Args:
            compounds: Dict[str, List[ParsedFile]] from IntegratedBrowserWidget
        """
        if not compounds:
            return None

        serialized = {}
        for compound_name, files_list in compounds.items():
            # files_list is a List[ParsedFile]
            serialized[compound_name] = {
                'name': compound_name,
                'files': []
            }

            # Serialize each ParsedFile
            for parsed_file in files_list:
                # Return dataclass as-is so NumpyEncoder can properly serialize with type markers
                # Do NOT use asdict() - it loses type info needed for deserialization!
                serialized[compound_name]['files'].append(parsed_file)

        return serialized

    @staticmethod
    def _deserialize_compounds(serialized_compounds: Optional[Dict]) -> Optional[Dict]:
        """
        Deserialize compound data from JSON storage back to ParsedFile format.

        Converts serialized compound data back to Dict[str, List[ParsedFile]].

        Args:
            serialized_compounds: Serialized compound data from JSON

        Returns:
            Dict[compound_name, List[ParsedFile]] suitable for IntegratedBrowserWidget
        """
        if not serialized_compounds:
            return None

        from data.file_parser import ParsedFile

        deserialized = {}
        for compound_name, compound_data in serialized_compounds.items():
            files_list = []

            # Extract files from nested structure
            if isinstance(compound_data, dict) and 'files' in compound_data:
                file_dicts = compound_data['files']
            elif isinstance(compound_data, list):
                # Fallback: directly a list of file dicts
                file_dicts = compound_data
            else:
                logger.warning(f"Unexpected compound data structure for {compound_name}")
                continue

            # Convert each file dict to ParsedFile dataclass
            for file_data in file_dicts:
                try:
                    # Check if it's a serialized dataclass with type marker
                    if isinstance(file_data, dict) and '__type__' in file_data:
                        if file_data['__type__'] == 'dataclass':
                            # Reconstruct dataclass using the generic reconstructor
                            parsed_file = SessionManager._deserialize_dataclass(
                                file_data['class'],
                                file_data['data']
                            )
                        else:
                            logger.warning(f"Unknown type marker: {file_data['__type__']}")
                            continue
                    elif isinstance(file_data, ParsedFile):
                        # Already a ParsedFile object (shouldn't happen, but handle it)
                        parsed_file = file_data
                    else:
                        # Old format: plain dict without type markers
                        # Handle Path objects
                        if 'file_path' in file_data:
                            file_data['file_path'] = Path(file_data['file_path'])
                        parsed_file = ParsedFile(**file_data)

                    files_list.append(parsed_file)
                except Exception as e:
                    logger.warning(f"Failed to deserialize file {file_data.get('file_path' if isinstance(file_data, dict) else None)}: {e}")
                    continue

            if files_list:
                deserialized[compound_name] = files_list

        return deserialized

    @staticmethod
    def _serialize_analysis(analysis_results: Optional[Dict]) -> Dict:
        """
        Serialize analysis results for JSON storage.

        Handles the structure from AnalysisWorker:
        {
            'kinetics_results': {compound_name: {...}},
            'statistics_results': {compound_name: {...}},
            'qy_results': {compound_name: {...}},
            'excluded_count': int
        }

        Also handles heterogeneous/surplus/vesicle results if present.
        """
        if not analysis_results:
            return {}

        serialized = {}

        # Iterate over top-level keys
        for key, value in analysis_results.items():
            # Handle simple values (like excluded_count)
            if not isinstance(value, dict):
                serialized[key] = value
                continue

            # Handle nested dict structures (kinetics_results, statistics_results, etc.)
            if value is None:
                serialized[key] = None
                continue

            serialized[key] = {}

            # Each compound's results
            for compound_name, compound_data in value.items():
                # compound_data can be:
                # - A dict with 'results' key (kinetics_results structure)
                # - A simple dict (statistics_results structure)
                # - A list of dataclasses (replicate results)
                # - A single dataclass

                if isinstance(compound_data, dict):
                    # Check if it has nested structure with 'results' key
                    if 'results' in compound_data:
                        # kinetics_results structure: {compound_name: {'results': [...], 'wavelength': ..., ...}}
                        serialized_compound = {}
                        for k, v in compound_data.items():
                            if k == 'results' and isinstance(v, list):
                                # Serialize list of KineticsResult dataclasses
                                serialized_compound[k] = [SessionManager._serialize_result(r) for r in v]
                            else:
                                # Simple values (wavelength, classification, etc.)
                                serialized_compound[k] = v
                        serialized[key][compound_name] = serialized_compound
                    else:
                        # Simple dict structure (statistics_results, qy_results)
                        serialized[key][compound_name] = SessionManager._serialize_result(compound_data)
                elif isinstance(compound_data, list):
                    # List of results
                    serialized[key][compound_name] = [
                        SessionManager._serialize_result(r) for r in compound_data
                    ]
                else:
                    # Single result
                    serialized[key][compound_name] = SessionManager._serialize_result(compound_data)

        return serialized

    @staticmethod
    def _serialize_result(result: Any) -> Any:
        """
        Serialize a single analysis result.

        Returns dataclasses as-is so NumpyEncoder can properly serialize them
        with type markers preserved. Do NOT use asdict() here - it loses type info!
        """
        # Just return as-is - NumpyEncoder will handle it during json.dump()
        return result

    @staticmethod
    def _deserialize_dataclass(class_name: str, data: Dict) -> Any:
        """
        Deserialize a dataclass from dict representation.

        Args:
            class_name: Name of the dataclass to reconstruct
            data: Dictionary of field values

        Returns:
            Reconstructed dataclass instance or dict if class not found
        """
        # Mapping of class names to class objects
        dataclass_registry = {
            # Core kinetics dataclasses
            'KineticsResult': KineticsResult,
            'FitParameters': FitParameters,
            'SNRResult': SNRResult,
            'FitQuality': FitQuality,
            'LiteratureModelResult': LiteratureModelResult,
            'WorkflowInfo': WorkflowInfo,

            # Heterogeneous dataclasses (both old and new versions)
            'HeterogeneousFitResult': HeteroFitResultNew,  # Use new version by default

            # Heterogeneous sub-dataclasses
            'VesicleGeometry': VesicleGeometry,
            'DiffusionParameters': DiffusionParameters,
            'SimulationResult': SimulationResult,

            # Surplus dataclasses
            'SurplusResult': SurplusResult,

            # Data parsing dataclasses
            'ParsedFile': ParsedFile,
        }

        # Get the dataclass type
        dataclass_type = dataclass_registry.get(class_name)

        if dataclass_type is None:
            logger.warning(f"Unknown dataclass type '{class_name}', returning as dict")
            return data

        try:
            logger.debug(f"Deserializing {class_name} with {len(data)} fields")

            # Recursively reconstruct nested dataclasses
            reconstructed_data = {}
            for field_name, field_value in data.items():
                if isinstance(field_value, dict) and '__type__' in field_value:
                    # This is a serialized nested object
                    if field_value['__type__'] == 'dataclass':
                        # Recursively deserialize nested dataclass
                        nested_class = field_value['class']
                        logger.debug(f"  Field '{field_name}': nested dataclass {nested_class}")
                        reconstructed_data[field_name] = SessionManager._deserialize_dataclass(
                            nested_class, field_value['data']
                        )
                    elif field_value['__type__'] == 'ndarray':
                        # NumPy array - reconstruct
                        logger.debug(f"  Field '{field_name}': numpy array {field_value['shape']}")
                        reconstructed_data[field_name] = np.array(
                            field_value['data'], dtype=field_value['dtype']
                        ).reshape(field_value['shape'])
                    else:
                        reconstructed_data[field_name] = field_value
                else:
                    reconstructed_data[field_name] = field_value

            # Create dataclass instance
            logger.debug(f"Successfully reconstructed {class_name}")
            return dataclass_type(**reconstructed_data)

        except Exception as e:
            logger.error(f"Failed to deserialize {class_name}: {e}", exc_info=True)
            logger.warning(f"Returning {class_name} as dict instead")
            return data

    @staticmethod
    def _reconstruct_analysis_results(analysis_data: Any) -> Any:
        """
        Recursively reconstruct dataclasses in analysis results.

        Handles the nested structure of analysis results where dataclasses
        are serialized with '__type__': 'dataclass' markers.

        Args:
            analysis_data: Analysis results data (dict, list, or other)

        Returns:
            Same structure with dataclasses reconstructed
        """
        if analysis_data is None:
            return None

        if isinstance(analysis_data, dict):
            # Check if this is a serialized dataclass
            if '__type__' in analysis_data and analysis_data['__type__'] == 'dataclass':
                # Reconstruct this dataclass
                return SessionManager._deserialize_dataclass(
                    analysis_data['class'],
                    analysis_data['data']
                )
            else:
                # Recursively process dict values
                return {
                    key: SessionManager._reconstruct_analysis_results(value)
                    for key, value in analysis_data.items()
                }

        elif isinstance(analysis_data, list):
            # Recursively process list items
            return [
                SessionManager._reconstruct_analysis_results(item)
                for item in analysis_data
            ]

        else:
            # Return as-is (primitive types, numpy arrays, etc.)
            return analysis_data

    @staticmethod
    def get_session_info(filepath: Path) -> Optional[Dict[str, str]]:
        """
        Get basic session information without loading full session.

        Useful for displaying session details in file picker dialog.

        Args:
            filepath: Path to session file

        Returns:
            Dictionary with keys: version, created, description, folder_path
            or None if file cannot be read
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            metadata = session_data.get('metadata', {})
            data = session_data.get('data', {})

            return {
                'version': metadata.get('version', 'Unknown'),
                'created': metadata.get('created', 'Unknown'),
                'description': metadata.get('description', ''),
                'folder_path': data.get('folder_path', 'Unknown'),
                'has_homogeneous': bool(session_data.get('analysis', {}).get('homogeneous')),
                'has_heterogeneous': bool(session_data.get('analysis', {}).get('heterogeneous')),
                'has_surplus': bool(session_data.get('analysis', {}).get('surplus')),
                'has_vesicle': bool(session_data.get('analysis', {}).get('vesicle'))
            }

        except Exception as e:
            logger.error(f"Failed to read session info: {e}")
            return None


# Convenience functions for GUI integration

def save_session_dialog(
    parent=None,
    initial_dir: Optional[Path] = None,
    **kwargs
) -> Optional[Path]:
    """
    Show save file dialog for session files.

    Returns:
        Selected filepath or None if cancelled
    """
    from PyQt6.QtWidgets import QFileDialog

    if initial_dir is None:
        initial_dir = Path.cwd()

    filepath, _ = QFileDialog.getSaveFileName(
        parent,
        "Save SOLIS Session",
        str(initial_dir / "session.solis.json"),
        "SOLIS Session Files (*.solis.json);;All Files (*.*)"
    )

    if filepath:
        filepath = Path(filepath)
        # Ensure .solis.json extension
        if not filepath.name.endswith('.solis.json'):
            if filepath.suffix == '.json':
                filepath = filepath.with_suffix('.solis.json')
            else:
                filepath = Path(str(filepath) + '.solis.json')
        return filepath

    return None


def load_session_dialog(
    parent=None,
    initial_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Show open file dialog for session files.

    Returns:
        Selected filepath or None if cancelled
    """
    from PyQt6.QtWidgets import QFileDialog

    if initial_dir is None:
        initial_dir = Path.cwd()

    filepath, _ = QFileDialog.getOpenFileName(
        parent,
        "Load SOLIS Session",
        str(initial_dir),
        "SOLIS Session Files (*.solis.json);;All Files (*.*)"
    )

    if filepath:
        return Path(filepath)

    return None


if __name__ == '__main__':
    print("SOLIS Session Manager")
    print("=" * 50)
    print("Provides session save/load functionality for SOLIS GUI")
    print("\nUsage:")
    print("  from utils.session_manager import SessionManager")
    print("  SessionManager.save_session(filepath, ...)")
    print("  session = SessionManager.load_session(filepath)")
