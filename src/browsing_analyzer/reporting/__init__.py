"""Reporting package: markdown report + Streamlit dashboard."""

from .dashboard import run_dashboard
from .generator import generate_markdown_report

__all__ = ["generate_markdown_report", "run_dashboard"]
