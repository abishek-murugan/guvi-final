"""Integration test: full pipeline on synthetic data."""

from __future__ import annotations

from browsing_analyzer.config import Settings
from browsing_analyzer.pipeline import run_pipeline


def test_full_pipeline_end_to_end(synthetic_data_files):
    history_path, ram_path = synthetic_data_files
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


def test_pipeline_rejects_invalid_window(tmp_path):
    settings = Settings()
    try:
        run_pipeline(settings=settings, window_days=99)
        raise AssertionError("Expected ValueError for invalid window")
    except ValueError:
        pass
