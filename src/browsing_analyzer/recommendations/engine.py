"""Rule-based + model-driven recommendation engine.

Every recommendation is *traceable*: it references the exact metric and
threshold that triggered it, so the reasoning chain is verifiable end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

SOCIAL = "Social Media"
ENTERTAINMENT = "Entertainment/Media"
LEARNING = "Learning/Education"


@dataclass
class Recommendation:
    """A single, evidence-backed recommendation."""

    title: str
    rationale: str
    evidence: str
    severity: str
    metric: str

    def to_dict(self) -> dict[str, str]:
        """Serialize for reporting and the dashboard."""
        return {
            "title": self.title,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "severity": self.severity,
            "metric": self.metric,
        }


def generate_recommendations(
    settings: Settings,
    sessions: pd.DataFrame,
    category_stats: pd.DataFrame,
    top_domains: pd.DataFrame,
    hour_stats: pd.DataFrame,
    dl_signals: dict[str, Any] | None = None,
) -> list[Recommendation]:
    """Generate traceable, actionable recommendations.

    Args:
        settings: Application settings (thresholds).
        sessions: Per-session feature table (``session_features``).
        category_stats: Category-level RAM aggregates.
        top_domains: Top domains with event counts.
        hour_stats: Hourly visit counts (``patterns["hourly"]``).
        dl_signals: Optional model-derived signals (predicted next category).

    Returns:
        List of :class:`Recommendation` instances.
    """
    recs: list[Recommendation] = []
    cfg = settings.recommendations
    late_night = cfg.late_night_hour

    if not sessions.empty:
        late_social = sessions[
            (sessions["session_start_hour"] >= late_night)
            & (sessions["primary_category"] == SOCIAL)
        ]
        if not late_social.empty:
            recs.append(
                Recommendation(
                    title="Reduce late-night social scrolling",
                    rationale="Sessions starting after 22:00 are dominated by social media, which is linked to poor sleep.",
                    evidence=(
                        f"{len(late_social)} of {len(sessions)} sessions start at hour >= {late_night} "
                        f"with {SOCIAL} as the primary category."
                    ),
                    severity="medium",
                    metric=f"session_start_hour >= {late_night} & primary_category == {SOCIAL}",
                )
            )

    if not category_stats.empty:
        top_cat = category_stats.sort_values("peak_used_mb", ascending=False)
        heavy = top_cat[top_cat["peak_used_mb"] >= cfg.category_ram_threshold_mb]
        if not heavy.empty:
            names = ", ".join(heavy["category"].tolist())
            recs.append(
                Recommendation(
                    title="Close memory-heavy tabs to reduce RAM pressure",
                    rationale="Some categories have a much larger memory footprint; closing idle tabs frees RAM.",
                    evidence=(
                        f"{names} peak at >= {cfg.category_ram_threshold_mb:.0f} MB of system RAM "
                        f"(top: {top_cat.iloc[0]['category']} @ {top_cat.iloc[0]['peak_used_mb']:.0f} MB)."
                    ),
                    severity="high",
                    metric=f"category peak_used_mb >= {cfg.category_ram_threshold_mb:.0f} MB",
                )
            )

    if not sessions.empty and "peak_used_mb" in sessions.columns:
        peaks = sessions["peak_used_mb"].dropna()
        if not peaks.empty:
            q90 = float(peaks.quantile(0.9))
            spike_threshold = max(cfg.ram_spike_threshold_mb, q90)
            spike_sessions = sessions[sessions["peak_used_mb"] >= spike_threshold]
            if not spike_sessions.empty:
                recs.append(
                    Recommendation(
                        title="Unusually high RAM during some sessions",
                        rationale="Peak RAM exceeds the 90th percentile of sessions; heavy tabs drive memory spikes.",
                        evidence=(
                            f"{len(spike_sessions)} of {len(sessions)} sessions hit RAM >= "
                            f"{spike_threshold:.0f} MB."
                        ),
                        severity="medium",
                        metric=f"peak_used_mb >= {spike_threshold:.0f} MB (p90)",
                    )
                )

    if not sessions.empty:
        avg_ram = sessions["mean_used_mb"].mean()
        if not pd.isna(avg_ram):
            recs.append(
                Recommendation(
                    title="Trim overall browser memory footprint",
                    rationale="Average session RAM is high; limiting open tabs reduces memory pressure.",
                    evidence=f"Average session RAM = {avg_ram:.0f} MB.",
                    severity="low",
                    metric=f"avg mean_used_mb = {avg_ram:.0f} MB",
                )
            )

    if not sessions.empty and "category_switches" in sessions.columns:
        high_switch = sessions[sessions["category_switches"] >= cfg.high_switch_threshold]
        share = len(high_switch) / len(sessions)
        if share > 0.15:
            recs.append(
                Recommendation(
                    title="Browsing is fragmented with rapid topic switching",
                    rationale="High switching rates suggest distraction; grouping related tasks improves focus.",
                    evidence=f"{share:.0%} of sessions switch categories >= {cfg.high_switch_threshold} times.",
                    severity="medium",
                    metric=f"category_switches >= {cfg.high_switch_threshold}",
                )
            )

    if not hour_stats.empty and "mean_visits" in hour_stats.columns:
        peak_hour = int(hour_stats["mean_visits"].idxmax())
        if peak_hour >= late_night:
            recs.append(
                Recommendation(
                    title="Peak browsing happens late at night",
                    rationale="Highest activity occurs after 22:00; shifting focused work to daytime may help.",
                    evidence=f"Peak hour = {peak_hour}:00.",
                    severity="low",
                    metric=f"peak hour = {peak_hour}",
                )
            )

    if not sessions.empty:
        entertainment = sessions[sessions["primary_category"] == ENTERTAINMENT]
        learning = sessions[sessions["primary_category"] == LEARNING]
        if len(learning) / len(sessions) < 0.2 and len(entertainment) / len(sessions) > 0.1:
            recs.append(
                Recommendation(
                    title="Shift some media time toward learning",
                    rationale="Media sessions outnumber learning sessions; a small reallocation supports skill growth.",
                    evidence=(
                        f"Learning share = {len(learning) / len(sessions):.0%} vs "
                        f"media share = {len(entertainment) / len(sessions):.0%}."
                    ),
                    severity="low",
                    metric=f"learning_share {len(learning) / len(sessions):.2f} < 0.2",
                )
            )

    if dl_signals and dl_signals.get("next_category"):
        nxt = dl_signals["next_category"]
        if nxt in {SOCIAL, ENTERTAINMENT}:
            recs.append(
                Recommendation(
                    title="Model predicts more social/media browsing ahead",
                    rationale="The LSTM ranks social/media as the most probable next category given recent sessions.",
                    evidence=(
                        f"Predicted next category = {nxt} "
                        f"(p={dl_signals.get('next_prob', 0.0):.2f})."
                    ),
                    severity="low",
                    metric=f"LSTM next = {nxt}",
                )
            )

    if not top_domains.empty:
        top_domain = top_domains.iloc[0]
        share = top_domain["event_count"] / top_domains["event_count"].sum()
        if share > 0.3:
            recs.append(
                Recommendation(
                    title=f"{top_domain['domain']} is your dominant site",
                    rationale="A single domain drives over 30% of visits; evaluate whether that time is intentional.",
                    evidence=f"{top_domain['domain']} = {share:.0%} of visits.",
                    severity="medium",
                    metric=f"{top_domain['domain']} share = {share:.2f}",
                )
            )

    if not recs:
        recs.append(
            Recommendation(
                title="Continue balanced browsing",
                rationale="No strong risk signals were detected in the current window.",
                evidence="All monitored metrics stayed within healthy thresholds.",
                severity="low",
                metric="baseline",
            )
        )

    logger.info("recommendations_generated", count=len(recs))
    return recs
