"""Data collection package.

The real-world collection pipeline (Chrome SQLite extraction, psutil RAM
logger) is described in the docstrings of each module. In this deliverable,
data is collected locally on the learner's machine and provided as CSV files
(``chrome_data.csv`` and ``ram_data.csv``); the collectors below ingest those
files and normalize them into the standard schemas used downstream.
"""

from .history import load_browsing_history
from .ram_logger import load_ram_log

__all__ = ["load_browsing_history", "load_ram_log"]
