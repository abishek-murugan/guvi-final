"""URL cleaning and domain extraction.

Privacy-first: every URL is reduced to its registrable domain. Query strings,
path fragments, and credentials are stripped so that only the domain and
derived metadata are stored in the sanitized dataset.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import pandas as pd
import tldextract

from ..utils.logging import get_logger

logger = get_logger(__name__)

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_FALLBACK_DOMAIN_RE = re.compile(r"^([a-zA-Z0-9.-]+)")

# Registered domains that are unlikely to be tracked by tldextract's snapshot
# are treated as non-external and labelled "local" for safety.
_LOCAL_SUFFIXES = {"localhost", "local", "internal"}


def strip_query_and_path(url: str) -> str:
    """Remove the scheme, query, fragment, and path from a URL.

    Returns the bare domain (or subdomain) portion.
    """
    if not url or not isinstance(url, str):
        return ""
    cleaned = url.strip()
    parsed = urlparse(cleaned if _SCHEME_RE.match(cleaned) else f"//{cleaned}")
    return parsed.netloc or cleaned


def extract_domain(url: str) -> str:
    """Extract the registrable domain from a URL.

    Falls back to a regex on the netloc when ``tldextract`` cannot identify a
    suffix, which keeps processing resilient for exotic test URLs.

    Args:
        url: A full or partial URL.

    Returns:
        Lowercased registrable domain, or empty string if unparseable.
    """
    netloc = strip_query_and_path(url).lower()
    if not netloc:
        return ""

    extracted = tldextract.extract(netloc)
    domain = (
        getattr(extracted, "top_domain_under_public_suffix", None)
        or extracted.registered_domain
        or ""
    )

    # Fallback: pull the last two labels of the host as a best-effort domain.
    if not domain and netloc not in _LOCAL_SUFFIXES:
        match = _FALLBACK_DOMAIN_RE.match(netloc)
        if match:
            labels = match.group(1).split(".")
            domain = ".".join(labels[-2:]) if len(labels) >= 2 else match.group(1)
    return domain or ""


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw browsing history DataFrame.

    Steps:
        * Drop rows with missing/empty URLs.
        * Drop exact duplicate ``(timestamp, url)`` rows.
        * Extract the registrable domain.
        * Add ``hour`` and ``date`` features from the timestamp.
        * Mark parse quality via the ``url_valid`` flag.

    Args:
        df: Raw history with ``timestamp`` and ``url`` columns.

    Returns:
        Cleaned DataFrame with ``domain``, ``hour``, ``date`` columns.
    """
    if df.empty:
        logger.warning("clean_history_empty_input")
        return df.copy()

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["url"])
    cleaned["url"] = cleaned["url"].astype(str).str.strip()
    cleaned = cleaned[cleaned["url"] != ""]

    cleaned["domain"] = cleaned["url"].map(extract_domain)
    cleaned["url_valid"] = cleaned["domain"] != ""

    before = len(df)
    cleaned = cleaned[cleaned["url_valid"]]
    cleaned = cleaned.drop_duplicates(subset=["timestamp", "url"]).reset_index(drop=True)

    cleaned["hour"] = cleaned["timestamp"].dt.hour
    cleaned["date"] = cleaned["timestamp"].dt.date.astype(str)
    cleaned["day_name"] = cleaned["timestamp"].dt.day_name()

    logger.info(
        "history_cleaned",
        dropped=before - len(cleaned),
        kept=len(cleaned),
        unique_domains=cleaned["domain"].nunique(),
    )
    return cleaned
