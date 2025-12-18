#!/usr/bin/env python3
"""
Logging configuration for SOLIS
Provides consistent logging across all modules
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Setup a logger with console and optional file output.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with detailed format (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default SOLIS configuration.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return setup_logger(name, level=logging.INFO)


# Module-level convenience function
def set_global_log_level(level: int):
    """
    Set logging level for all SOLIS loggers.

    Args:
        level: logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR
    """
    logging.getLogger('kinetics_analyzer').setLevel(level)
    logging.getLogger('statistical_analyzer').setLevel(level)
    logging.getLogger('quantum_yield_calculator').setLevel(level)
    logging.getLogger('file_parser').setLevel(level)
    logging.getLogger('csv_exporter').setLevel(level)
    logging.getLogger('plotly_plotter').setLevel(level)
