"""Unit tests for URL cleaning and domain extraction."""

from __future__ import annotations

import pandas as pd

from browsing_analyzer.prep.cleaner import clean_history, extract_domain, strip_query_and_path


def test_extract_domain_simple():
    assert extract_domain("https://www.facebook.com/feed?ref=1") == "facebook.com"


def test_extract_domain_strips_query_and_path():
    assert extract_domain("https://stackoverflow.com/questions/123") == "stackoverflow.com"


def test_extract_domain_handles_bare_host():
    assert extract_domain("github.com") == "github.com"


def test_extract_domain_empty_for_invalid():
    assert extract_domain("") == ""


def test_strip_query_and_path_removes_query_string():
    url = "https://example.com/search?q=secret&token=abc#section"
    assert strip_query_and_path(url) == "example.com"


def test_strip_query_and_path_handles_missing_scheme():
    assert strip_query_and_path("example.com/path?x=1") == "example.com"


def test_clean_history_adds_domain_and_derived_columns(sample_history_df):
    cleaned = clean_history(sample_history_df)
    assert "domain" in cleaned.columns
    assert "hour" in cleaned.columns
    assert "date" in cleaned.columns
    assert "day_name" in cleaned.columns
    assert cleaned["url_valid"].all()
    assert cleaned["domain"].iloc[0] == "facebook.com"


def test_clean_history_drops_duplicates(sample_history_df):
    df = pd.concat([sample_history_df, sample_history_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_history(df)
    assert len(cleaned) == len(sample_history_df)


def test_clean_history_drops_invalid_urls(sample_history_df):
    df = pd.concat(
        [
            sample_history_df,
            pd.DataFrame(
                [{"timestamp": pd.Timestamp("2026-07-28 15:00:00"), "url": "", "title": "empty"}]
            ),
        ],
        ignore_index=True,
    )
    cleaned = clean_history(df)
    assert len(cleaned) == len(sample_history_df)
