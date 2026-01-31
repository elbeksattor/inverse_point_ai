"""
Logging utility for IP AI v3.

Uses structlog for structured, JSON-friendly logging.
"""

import logging
import sys
from typing import Optional

import structlog


def setup_logger(
    name: str = "ip_ai_v3",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    json_format: bool = False
) -> structlog.BoundLogger:
    """
    Set up structured logging.

    Args:
        name: Logger name
        level: Logging level (INFO, DEBUG, etc.)
        log_file: Optional file path for logging
        json_format: Use JSON output format

    Returns:
        Configured structlog logger
    """
    # Configure standard logging
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=handlers,
        force=True
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger(name)


def get_logger(name: str = "ip_ai_v3") -> structlog.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return structlog.get_logger(name)
