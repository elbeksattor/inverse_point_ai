#!/usr/bin/env python3
"""
Logging utility for IP AI Analytics
Provides structured logging with different levels and outputs
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import structlog

def setup_logger(
    name: str = "ip_ai_analytics",
    log_level: str = "INFO",
    log_dir: str = "output/logs",
    console_output: bool = True,
    file_output: bool = True
) -> structlog.BoundLogger:
    """
    Set up structured logger with console and file outputs

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console_output: Enable console logging
        file_output: Enable file logging

    Returns:
        Configured structlog logger
    """
    # Create log directory
    if file_output:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure Python logging
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        handlers=[]
    )

    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        logging.root.addHandler(console_handler)

    # Add file handler
    if file_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        logging.root.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger(name)
    logger.info("logger_initialized", log_level=log_level, log_dir=log_dir)

    return logger


def get_logger(name: str = "ip_ai_analytics") -> structlog.BoundLogger:
    """
    Get existing logger instance

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return structlog.get_logger(name)
