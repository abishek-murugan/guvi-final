"""Command-line interface for the browsing analyzer.

Usage (installed as ``browsing-analyzer``):

.. code-block:: bash

    browsing-analyzer pipeline --window 4
    browsing-analyzer preprocess --window 4
    browsing-analyzer train --window 4
    browsing-analyzer report --window 4
    browsing-analyzer dashboard
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from .config import Settings, load_settings
from .pipeline import run_pipeline
from .reporting.generator import generate_markdown_report
from .utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

app = typer.Typer(name="browsing-analyzer", help="Analyze browsing history & RAM correlation.")

WINDOW_OPTION = Annotated[
    int | None,
    typer.Option("--window", "-w", help="Analysis window in days (3, 4 or 5)."),
]


def _settings(window: int | None) -> Settings:
    settings = load_settings()
    if window is not None and window not in settings.time_windows:
        raise typer.BadParameter(f"--window must be one of {settings.time_windows}")
    return settings


@app.command()
def pipeline(window: WINDOW_OPTION = None) -> None:
    """Run the full pipeline: collect -> preprocess -> analyze -> train -> recommend."""
    configure_logging()
    settings = _settings(window)
    result = run_pipeline(settings=settings, window_days=window)
    report_path = Path(settings.output.report_path) / settings.output.report_filename
    generate_markdown_report(result, report_path)
    typer.echo(f"Pipeline complete. Report written to {report_path}")
    typer.echo(f"Events: {len(result.events)} | Sessions: {len(result.sessions)}")
    typer.echo(
        f"Test accuracy: {result.dl_result.test_accuracy:.3f} "
        f"(baseline {result.dl_result.baseline_accuracy:.3f})"
        if result.dl_result
        else "Model skipped."
    )


@app.command()
def preprocess(window: WINDOW_OPTION = None) -> None:
    """Preprocess and sessionize data only."""
    configure_logging()
    settings = _settings(window)
    result = run_pipeline(settings=settings, window_days=window, train_model_flag=False)
    out = Path(settings.data.processed_dir)
    typer.echo(f"Processed {len(result.events)} events / {len(result.sessions)} sessions -> {out}")


@app.command()
def train(window: WINDOW_OPTION = None) -> None:
    """Train the LSTM next-category model."""
    configure_logging()
    settings = _settings(window)
    result = run_pipeline(settings=settings, window_days=window, train_model_flag=True)
    if result.dl_result is None:
        raise typer.Exit(code=1)
    dl = result.dl_result
    typer.echo(f"Accuracy: {dl.test_accuracy:.3f} | Macro F1: {dl.macro_f1:.3f}")
    typer.echo(f"Baseline accuracy: {dl.baseline_accuracy:.3f} | Baseline F1: {dl.baseline_f1:.3f}")
    typer.echo("Confusion matrix:")
    labels = sorted(set(result.events["category"]))
    for i, label in enumerate(labels):
        typer.echo(f"  {label:>12}: " + " ".join(f"{v:3d}" for v in dl.confusion[i]))


@app.command()
def report(window: WINDOW_OPTION = None) -> None:
    """Generate the markdown report."""
    configure_logging()
    settings = _settings(window)
    result = run_pipeline(settings=settings, window_days=window)
    path = Path(settings.output.report_path) / settings.output.report_filename
    generate_markdown_report(result, path)
    typer.echo(f"Report written to {path}")


@app.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    configure_logging()
    settings = load_settings()
    port = settings.output.dashboard_port
    module = "browsing_analyzer.reporting.dashboard"
    cmd = [sys.executable, "-m", "streamlit", "run", "-m", module, "--server.port", str(port)]
    typer.echo(f"Launching dashboard on http://localhost:{port}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    app()
