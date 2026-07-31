"""Integration test: full pipeline on synthetic data."""

from __future__ import annotations

from pathlib import Path

from browsing_analyzer.config import Settings
from browsing_analyzer.pipeline import run_pipeline

SYNTH_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "generate_synthetic_data.py"


def _synthetic_files(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a small synthetic dataset for pipeline testing."""
    import subprocess
    import sys

    out = tmp_path / "raw"
    subprocess.run(
        [sys.executable, str(SYNTH_SCRIPT), "--days", "3", "--seed", "1", "--out", str(out)],
        check=True,
        capture_output=True,
    )
    return out / "chrome_data.csv", out / "ram_data.csv"


def test_full_pipeline_end_to_end(tmp_path: Path):
    history_path, ram_path = _synthetic_files(tmp_path)
    settings = Settings()
    result = run_pipeline(
        settings=settings,
        window_days=3,
        history_path=history_path,
        ram_path=ram_path,
        train_model_flag=True,
    )

    assert len(result.events) > 0
    assert len(result.sessions) > 0
    assert result.cluster is not None
    assert result.dl_result is not None
    assert len(result.recommendations) >= 5
    assert "category" in result.events.columns

    # Privacy: raw URLs must be dropped from processed outputs.
    assert "url" not in result.events.columns


def test_pipeline_rejects_invalid_window(tmp_path: Path):
    settings = Settings()
    try:
        run_pipeline(settings=settings, window_days=99)
        raise AssertionError("Expected ValueError for invalid window")
    except ValueError:
        pass
