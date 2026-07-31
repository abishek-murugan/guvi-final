#!/usr/bin/env python
"""One-shot pipeline runner.

Example:
    uv run python scripts/run_pipeline.py --window 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from browsing_analyzer.config import load_settings
from browsing_analyzer.pipeline import run_pipeline
from browsing_analyzer.reporting.generator import generate_markdown_report
from browsing_analyzer.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full browsing-analysis pipeline.")
    parser.add_argument("--window", type=int, default=None, choices=[3, 4, 5])
    parser.add_argument("--history", type=Path, default=None, help="Chrome history CSV.")
    parser.add_argument("--ram", type=Path, default=None, help="RAM log CSV.")
    parser.add_argument("--no-train", action="store_true", help="Skip LSTM training.")
    args = parser.parse_args()

    configure_logging()
    settings = load_settings()
    result = run_pipeline(
        settings=settings,
        window_days=args.window,
        history_path=args.history,
        ram_path=args.ram,
        train_model_flag=not args.no_train,
    )

    report_path = Path(settings.output.report_path) / settings.output.report_filename
    generate_markdown_report(result, report_path)

    logger.info(
        "run_complete",
        events=len(result.events),
        sessions=len(result.sessions),
        report=str(report_path),
    )
    print(f"Report: {report_path}")
    print(f"Events: {len(result.events)} | Sessions: {len(result.sessions)}")


if __name__ == "__main__":
    main()
