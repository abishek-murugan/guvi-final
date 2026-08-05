"""Domain -> category mapping.

The 654-entry mapping lives in the notebook (``notebooks/guvi-final.ipynb``,
the ``domain_to_category_mapping`` literal). It is parsed from the notebook at
runtime so the pipeline can never drift from the notebook, and cached to YAML
for inspection and offline use. Unknown domains map to ``Other``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)


def _load_from_notebook(notebook_path: Path) -> dict[str, str]:
    """Extract the ``domain_to_category_mapping`` dict from the notebook."""
    cells = json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "domain_to_category_mapping" in source:
            namespace: dict[str, Any] = {}
            exec(source, namespace)
            return dict(namespace["domain_to_category_mapping"])
    raise RuntimeError(f"No domain mapping found in notebook: {notebook_path}")


def _to_yaml(mapping: dict[str, str]) -> dict[str, list[str]]:
    """Group the flat mapping by category (domain lists per category)."""
    grouped: dict[str, list[str]] = {}
    for domain, category in mapping.items():
        grouped.setdefault(category, []).append(domain)
    return grouped


class Categorizer:
    """Assigns a category to a domain using the notebook's mapping.

    Args:
        notebook_path: Path to the source notebook.
        cache_path: Optional YAML path to write the extracted mapping.
    """

    def __init__(self, notebook_path: Path, cache_path: Path | None = None) -> None:
        self._mapping = _load_from_notebook(notebook_path)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                yaml.safe_dump(_to_yaml(self._mapping), sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        logger.info("categorizer_loaded", mappings=len(self._mapping))

    def categorize(self, domain: str) -> str | float:
        """Return the category for a domain (``Other`` when unknown)."""
        if pd.isna(domain):
            return np.nan
        return self._mapping.get(domain, "Other")

    def categorize_series(self, domains: pd.Series) -> pd.Series:
        """Apply :meth:`categorize` to a pandas Series of domains."""
        return domains.map(self.categorize)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the mapping as a two-column DataFrame."""
        rows = sorted(self._mapping.items())
        return pd.DataFrame(rows, columns=["domain", "category"])
