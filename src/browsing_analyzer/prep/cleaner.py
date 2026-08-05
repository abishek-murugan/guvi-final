"""URL cleaning and feature extraction.

Replicates the notebook's ``clean_and_extract_url_parts``: query strings and
fragments are stripped from each URL and the bare domain (netloc without a
leading ``www.``) is extracted into a dedicated column.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def clean_and_extract_url_parts(url: str) -> tuple[str, str | float]:
    """Return ``(cleaned_url, domain)`` for a raw URL.

    The cleaned URL drops the query string and fragment; the domain is the
    netloc without a leading ``www.``. On parsing failure the original URL is
    returned with a NaN domain.
    """
    try:
        parsed_url = urlsplit(url)
        cleaned_url = parsed_url._replace(query="", fragment="").geturl()
        domain = parsed_url.netloc.replace("www.", "")
        return cleaned_url, domain
    except Exception:
        return url, np.nan


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw browsing history DataFrame.

    Steps:
        * Strip query strings / fragments from ``url``.
        * Extract the bare ``domain``.
        * Convert ``timestamp`` to datetime and add ``hour``, ``date`` and
          ``day_name`` features.

    Args:
        df: Raw history with ``timestamp`` and ``url`` columns.

    Returns:
        Cleaned DataFrame with cleaned ``url``, ``domain`` and time features.
    """
    cleaned = df.copy()
    cleaned[["url", "domain"]] = cleaned["url"].apply(
        lambda x: pd.Series(clean_and_extract_url_parts(x))
    )
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"])
    cleaned["hour"] = cleaned["timestamp"].dt.hour
    cleaned["date"] = cleaned["timestamp"].dt.date
    cleaned["day_name"] = cleaned["timestamp"].dt.day_name()
    logger.info("history_cleaned", rows=len(cleaned))
    return cleaned
