"""Analytics package: RAM correlation, clustering, temporal patterns."""

from .clustering import BehaviorClusterer
from .patterns import discover_time_patterns
from .ram_correlation import align_ram_with_events, session_ram_stats

__all__ = [
    "align_ram_with_events",
    "session_ram_stats",
    "BehaviorClusterer",
    "discover_time_patterns",
]
