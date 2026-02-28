"""Centralized logging configuration for DocuMind-RAG."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a consistent format. Safe to call once at startup."""
    level_value = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level_value,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
