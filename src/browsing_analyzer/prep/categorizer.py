"""Domain -> category mapping.

Loads the curated ``config/domain_categories.yaml`` table and assigns each
domain a behavioral category. Unknown domains are classified with a small set
of keyword heuristics, falling back to ``other``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CATEGORIES_YAML = Path(__file__).resolve().parents[3] / "config" / "domain_categories.yaml"


def _normalize_domain(domain: str) -> str:
    """Lowercase a domain and drop a leading ``www.`` subdomain."""
    lowered = domain.strip().lower()
    if lowered.startswith("www."):
        return lowered.removeprefix("www.")
    return lowered


# Keyword -> category heuristics used when the curated table has no entry.
_KEYWORD_MAP: dict[str, str] = {
    "social": "social",
    "facebook": "social",
    "instagram": "social",
    "reddit": "social",
    "discord": "social",
    "twitch": "media",
    "youtube": "media",
    "video": "media",
    "learn": "learning",
    "tutorial": "learning",
    "course": "learning",
    "docs": "learning",
    "edu": "learning",
    "school": "learning",
    "shop": "shopping",
    "store": "shopping",
    "cart": "shopping",
    "news": "news",
    "finance": "finance",
    "bank": "finance",
    "invest": "finance",
    "health": "health",
    "med": "health",
    "travel": "travel",
    "flight": "travel",
    "hotel": "travel",
    "game": "entertainment",
    "movie": "entertainment",
    "film": "entertainment",
    "sport": "entertainment",
    "search": "search",
    "google": "search",
    "mail": "productivity",
    "drive": "productivity",
}


class Categorizer:
    """Assigns a category to a domain using a curated mapping + heuristics.

    Args:
        categories_path: Path to the domain category YAML. Defaults to the
            repository ``config/domain_categories.yaml``.
    """

    def __init__(self, categories_path: Path | None = None) -> None:
        self._path = categories_path or DEFAULT_CATEGORIES_YAML
        with self._path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        self._table: dict[str, str] = {}
        for category, domains in raw.items():
            for domain in domains:
                if isinstance(domain, str) and domain:
                    self._table[_normalize_domain(domain)] = category
        logger.info("categorizer_loaded", mappings=len(self._table))

    def categorize(self, domain: str) -> str:
        """Return the category for a domain.

        Priority: exact match in curated table -> keyword heuristic -> ``other``.
        """
        key = _normalize_domain(domain or "")
        if not key:
            return "other"

        if key in self._table:
            return self._table[key]

        for keyword, category in _KEYWORD_MAP.items():
            if keyword in key:
                return category
        return "other"

    def categorize_series(self, domains: pd.Series) -> pd.Series:
        """Apply :meth:`categorize` to a pandas Series of domains."""
        return domains.map(self.categorize)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the curated mapping as a DataFrame (for reporting/debug)."""
        rows = [{"domain": d, "category": c} for d, c in sorted(self._table.items())]
        return pd.DataFrame(rows, columns=["domain", "category"])
