"""Time helpers shared across the package."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np


def to_local_datetime(ts: str | float | int | None) -> datetime:
    """Convert a timestamp (ISO string or Unix epoch) to local time.

    Args:
        ts: ISO-format string or numeric Unix timestamp (seconds).

    Returns:
        A naive local datetime.
    """
    if ts is None:
        raise ValueError("Cannot parse None timestamp")
    if isinstance(ts, str):
        parsed = datetime.fromisoformat(ts)
        if parsed.tzinfo is not None:
            return parsed.astimezone().replace(tzinfo=None)
        return parsed
    return datetime.fromtimestamp(float(ts)).astimezone().replace(tzinfo=None)


def as_epoch(dt: datetime) -> float:
    """Return Unix epoch (seconds) for a datetime, assuming local time."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def round_to_minute(dt: datetime) -> datetime:
    """Round a datetime down to the nearest minute."""
    return dt.replace(second=0, microsecond=0)


def date_range(start: datetime, end: datetime, days: int) -> list[datetime]:
    """Return a list of datetimes covering the window ``[start, end]``."""
    return [start + timedelta(days=i) for i in range(days)]


def hour_bucket(series: np.ndarray) -> np.ndarray:
    """Extract integer hour buckets from a datetime array."""
    return np.array([dt.hour for dt in series], dtype=np.int32)
