"""
logger_setup.py

Provides centralized logging configuration for Edge Testing modules.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file=None, level=logging.INFO, max_size=5*1024*1024, backup_count=3):
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        log_file: Path to the log file (if None, only console logging is enabled)
        level: Logging level (default: INFO)
        max_size: Max size of log file before rotation in bytes (default: 5MB)
        backup_count: Number of backup files to keep (default: 3)
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Ensure output directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_size, 
            backupCount=backup_count
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    return logger
