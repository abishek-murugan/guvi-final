"""Structured logging helpers built on :mod:`structlog`.

The loggers emit JSON records to a rotating file and optionally to the
console. Configure once via :func:`configure_logging`, then retrieve a
logger anywhere with :func:`get_logger`.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger

_EVENT_PROCESSORS: list = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def configure_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure structlog for both console and file output.

    Args:
        level: Minimum log level (e.g. ``"INFO"``, ``"DEBUG"``).
        log_dir: Directory where rotating log files are written.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    console_renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    file_renderer = structlog.processors.JSONRenderer()

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_path / "browsing_analyzer.log",
                maxBytes=5_000_000,
                backupCount=3,
            ),
        ],
    )

    # File handler uses JSON, console handler uses pretty rendering.
    _logging = logging.getLogger()
    for handler in _logging.handlers:
        processors = [*_EVENT_PROCESSORS]
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            processors.append(file_renderer)
        else:
            processors.append(console_renderer)
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        break


def get_logger(name: str = "browsing_analyzer") -> FilteringBoundLogger:
    """Return a bound structlog logger for the given module name.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        A structured, bound logger.
    """
    logger = structlog.get_logger(name)
    return cast(FilteringBoundLogger, logger)
