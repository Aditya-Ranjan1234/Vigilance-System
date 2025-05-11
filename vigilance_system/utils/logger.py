"""
Logging utility for the vigilance system.

This module provides a configured logger for consistent logging across the system.
"""

import os
import sys
from datetime import datetime
from loguru import logger
from typing import Optional

from vigilance_system.utils.config import config


def setup_logger(log_level: Optional[str] = None) -> None:
    """
    Configure the logger with settings from the configuration file.

    Args:
        log_level: Override the log level from config if provided
    """
    # Get logging configuration
    level = log_level or config.get('logging.level', 'INFO')
    log_file = config.get('logging.file', 'logs/vigilance.log')
    max_size = config.get('logging.max_size_mb', 10) * 1024 * 1024  # Convert to bytes
    backup_count = config.get('logging.backup_count', 5)

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=max_size,
        retention=backup_count,
        compression="zip"
    )

    logger.info(f"Logger initialized at level {level}")


def get_logger(name: str):
    """
    Get a logger with the given name.

    Args:
        name: The name for the logger, typically the module name

    Returns:
        Logger: A configured logger instance
    """
    return logger.bind(name=name)


# Initialize logger when module is imported
setup_logger()
