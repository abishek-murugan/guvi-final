"""Command-line interface for the browsing analyzer.

Usage (installed as ``browsing-analyzer``):

.. code-block:: bash

    browsing-analyzer pipeline
    browsing-analyzer preprocess
    browsing-analyzer train
    browsing-analyzer report
"""

from __future__ import annotations

from pathlib import Path

import typer

from .config import load_settings
from .pipeline import load_pipeline_result, run_pipeline
from .reporting.generator import generate_markdown_report
from .utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

app = typer.Typer(name="browsing-analyzer", help="Analyze browsing history & RAM correlation.")


@app.command()
def pipeline() -> None:
    """Run the full pipeline and regenerate all artifacts."""
    configure_logging()
    settings = load_settings()
    result = run_pipeline(settings)
    report_path = Path(settings.output.report_path) / settings.output.report_filename
    generate_markdown_report(result, report_path)
    _print_summary(result, report_path)


@app.command()
def preprocess() -> None:
    """Preprocess and sessionize data only (no model training)."""
    configure_logging()
    settings = load_settings()
    result = run_pipeline(settings, train_model_flag=False)
    typer.echo(
        f"Processed {len(result.events)} events / {len(result.session_features)} sessions "
        f"-> {settings.data.processed_dir}"
    )


@app.command()
def train() -> None:
    """Train the LSTM next-category model and save artifacts."""
    configure_logging()
    settings = load_settings()
    result = run_pipeline(settings, train_model_flag=True)
    if result.dl_result is None:
        raise typer.Exit(code=1)
    typer.echo(f"Test accuracy: {result.dl_result.test_accuracy:.3f}")
    typer.echo(f"Model saved to {Path(settings.data.models_dir) / settings.data.model_output_file}")


@app.command()
def report() -> None:
    """Generate the markdown report from persisted artifacts."""
    configure_logging()
    settings = load_settings()
    result = load_pipeline_result(settings)
    report_path = Path(settings.output.report_path) / settings.output.report_filename
    generate_markdown_report(result, report_path)
    typer.echo(f"Report written to {report_path}")


def _print_summary(result, report_path: Path) -> None:
    """Print a concise pipeline summary."""
    typer.echo(f"Events: {len(result.events)} | Sessions: {len(result.session_features)}")
    if result.dl_result is not None:
        typer.echo(f"Test accuracy: {result.dl_result.test_accuracy:.3f}")
    typer.echo(f"Report written to {report_path}")


if __name__ == "__main__":
    app()
