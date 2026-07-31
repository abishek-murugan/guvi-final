"""Unit tests for domain categorization."""

from __future__ import annotations

import pandas as pd

from browsing_analyzer.prep.categorizer import Categorizer


def test_categorize_exact_mapping():
    cat = Categorizer()
    assert cat.categorize("facebook.com") == "social"
    assert cat.categorize("www.youtube.com") == "media"
    assert cat.categorize("github.com") == "learning"
    assert cat.categorize("amazon.in") == "shopping"


def test_categorize_www_prefix_handled():
    cat = Categorizer()
    assert cat.categorize("www.reddit.com") == "social"


def test_categorize_unknown_falls_back_to_other():
    cat = Categorizer()
    assert cat.categorize("randomblog.xyz") == "other"


def test_categorize_keyword_heuristic():
    cat = Categorizer()
    # 'course' keyword should route to learning.
    assert cat.categorize("mycourses.example.com") == "learning"


def test_categorize_series(sample_history_df):
    cat = Categorizer()
    domains = pd.Series(["facebook.com", "youtube.com", "stackoverflow.com"])
    result = cat.categorize_series(domains)
    assert result.tolist() == ["social", "media", "learning"]


def test_to_dataframe_returns_mapping():
    cat = Categorizer()
    df = cat.to_dataframe()
    assert {"domain", "category"}.issubset(df.columns)
    assert len(df) > 0
