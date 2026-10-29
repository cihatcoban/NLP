"""
Structured logging configuration.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "nlp_connotation",
    level: str = "INFO",
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configures and returns a structured logger.
    Avoids duplicate handlers when called multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        fmt = log_format or "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
