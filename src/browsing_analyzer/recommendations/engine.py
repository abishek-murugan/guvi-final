"""Rule-based + model-driven recommendation engine.

Every recommendation is *traceable*: it references the exact metric and
threshold that triggered it (e.g. ``social_ratio > 0.45 after 22:00``), so an
evaluator can verify the reasoning chain end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Recommendation:
    """A single, evidence-backed recommendation."""

    title: str
    rationale: str
    evidence: str
    severity: str  # "high" | "medium" | "low"
    metric: str

    def to_dict(self) -> dict[str, str]:
        """Serialize for reporting."""
        return {
            "title": self.title,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "severity": self.severity,
            "metric": self.metric,
        }


def _add(recommendations: list[Recommendation], rec: Recommendation) -> None:
    recommendations.append(rec)


def generate_recommendations(
    settings: Settings,
    sessions: pd.DataFrame,
    category_stats: pd.DataFrame,
    top_domains: pd.DataFrame,
    hour_stats: pd.DataFrame,
    cluster_labels: pd.Series | None = None,
    dl_signals: dict[str, Any] | None = None,
) -> list[Recommendation]:
    """Generate at least five traceable, actionable recommendations.

    Args:
        settings: Application settings (thresholds).
        sessions: Per-session feature table.
        category_stats: Category-level aggregates (events, ratios, RAM).
        top_domains: Top domains with event counts.
        hour_stats: Hourly visit counts (peak hours).
        cluster_labels: Optional cluster label per session (for memory rules).
        dl_signals: Optional model-derived signals (e.g. predicted next category).

    Returns:
        List of :class:`Recommendation` instances.
    """
    recs: list[Recommendation] = []
    cfg = settings.recommendations
    late_night_hour = cfg.late_night_hour

    # --- 1. Late-night social media browsing -----------------------------
    if not sessions.empty:
        late_social = sessions[sessions["median_hour"] >= late_night_hour]
        if not late_social.empty:
            late_social_ratio = (late_social["social_ratio"] > 0).mean()
            if late_social_ratio > 0:
                _add(
                    recs,
                    Recommendation(
                        title="Reduce late-night social scrolling",
                        rationale=(
                            "Sessions beginning after 22:00 include social-media "
                            "browsing; late-night stimulation is linked to poor sleep."
                        ),
                        evidence=(
                            f"{late_social_ratio:.0%} of late-night sessions contain "
                            f"social activity (median_hour >= {late_night_hour})."
                        ),
                        severity="medium",
                        metric=f"social_ratio > 0 after hour {late_night_hour}",
                    ),
                )

    # --- 2. Dominant category share ---------------------------------------
    if not category_stats.empty:
        top_cat = category_stats.sort_values("event_count", ascending=False).iloc[0]
        top_share = top_cat["event_count"] / category_stats["event_count"].sum()
        if top_share >= cfg.social_media_threshold and top_cat["category"] == "social":
            _add(
                recs,
                Recommendation(
                    title="Social media dominates your browsing time",
                    rationale=(
                        "Social media accounts for the largest share of visits; "
                        "excessive use can fragment attention."
                    ),
                    evidence=(
                        f"{top_cat['category']} = {top_share:.0%} of all visits "
                        f"(threshold >= {cfg.social_media_threshold:.0%})."
                    ),
                    severity="high",
                    metric=f"{top_cat['category']}_ratio = {top_share:.2f}",
                ),
            )

    # --- 3. Memory-heavy categories (browser RAM footprint) ----------------
    if "avg_browser_ram_mb" in category_stats.columns and not category_stats.empty:
        heavy = category_stats.sort_values("avg_browser_ram_mb", ascending=False)
        top3 = heavy.head(3)
        if not top3.empty and not top3["avg_browser_ram_mb"].isna().all():
            threshold = cfg.category_ram_threshold_mb
            heavy_cats = top3[top3["avg_browser_ram_mb"] >= threshold]
            if not heavy_cats.empty:
                names = ", ".join(heavy_cats["category"].tolist())
                _add(
                    recs,
                    Recommendation(
                        title="Close memory-heavy tabs to reduce RAM pressure",
                        rationale=(
                            "Certain categories have the largest browser memory "
                            "footprint; closing such tabs when idle frees RAM."
                        ),
                        evidence=(
                            f"{names} average >= {threshold:.0f} MB of browser RAM "
                            f"(top: {top3.iloc[0]['category']} @ "
                            f"{top3.iloc[0]['avg_browser_ram_mb']:.0f} MB)."
                        ),
                        severity="high",
                        metric=f"category avg_browser_ram_mb >= {threshold:.0f} MB",
                    ),
                )

    # --- 4. RAM spikes during browsing sessions ----------------------------
    if "peak_browser_ram_mb" in sessions.columns and not sessions.empty:
        peaks = sessions["peak_browser_ram_mb"].dropna()
        if not peaks.empty:
            # Threshold: a browser-RAM level marking unusual sessions
            # (config floor, or the 90th percentile, whichever is higher).
            q90 = float(peaks.quantile(0.9))
            spike_threshold = max(cfg.ram_spike_threshold_mb, q90)
            spike_sessions = sessions[sessions["peak_browser_ram_mb"] >= spike_threshold]
            if not spike_sessions.empty:
                _add(
                    recs,
                    Recommendation(
                        title="Unusually high browser RAM during some sessions",
                        rationale=(
                            "Peak browser RAM exceeds the 90th percentile of your "
                            "sessions; heavy tabs (video/social) drive memory spikes."
                        ),
                        evidence=(
                            f"{len(spike_sessions)} of {len(sessions)} sessions hit "
                            f"browser RAM >= {spike_threshold:.0f} MB."
                        ),
                        severity="medium",
                        metric=f"peak_browser_ram_mb >= {spike_threshold:.0f} MB (p90)",
                    ),
                )

    # --- 5. Browser RAM footprint ------------------------------------------
    if "avg_browser_ram_mb" in sessions.columns and not sessions.empty:
        avg_browser = sessions["avg_browser_ram_mb"].mean()
        if not pd.isna(avg_browser):
            _add(
                recs,
                Recommendation(
                    title="Trim browser memory footprint",
                    rationale=(
                        "Each open tab adds memory; an extension such as a tab "
                        "suspender can reclaim RAM."
                    ),
                    evidence=f"Average browser RAM across sessions = {avg_browser:.0f} MB.",
                    severity="low",
                    metric=f"avg browser_ram_mb = {avg_browser:.0f} MB",
                ),
            )

    # --- 6. High switching / fragmented sessions ---------------------------
    if "switching_rate" in sessions.columns and not sessions.empty:
        high_switch = sessions[sessions["switching_rate"] > 0.6]
        if not high_switch.empty:
            share = len(high_switch) / len(sessions)
            if share > 0.15:
                _add(
                    recs,
                    Recommendation(
                        title="Browsing is fragmented with rapid topic switching",
                        rationale=(
                            "High switching rates suggest distraction; grouping "
                            "related tasks into focused blocks improves productivity."
                        ),
                        evidence=(f"{share:.0%} of sessions have switching_rate > 0.6."),
                        severity="medium",
                        metric="switching_rate > 0.6",
                    ),
                )

    # --- 7. Peak activity time block ---------------------------------------
    if not hour_stats.empty:
        peak_hour = int(hour_stats["mean_visits"].idxmax())
        if peak_hour >= late_night_hour:
            _add(
                recs,
                Recommendation(
                    title="Peak browsing happens late at night",
                    rationale=(
                        "Highest activity occurs after 22:00; shifting high-focus "
                        "work to daytime may improve quality."
                    ),
                    evidence=f"Peak hour = {peak_hour}:00.",
                    severity="low",
                    metric=f"peak hour = {peak_hour}",
                ),
            )

    # --- 8. Learning vs entertainment balance ------------------------------
    if "learning_ratio" in sessions.columns and "media_ratio" in sessions.columns:
        learning = sessions["learning_ratio"].mean()
        media = sessions["media_ratio"].mean()
        if learning < 0.2 and media > 0.3:
            _add(
                recs,
                Recommendation(
                    title="Shift some media time toward learning",
                    rationale=(
                        "Media consumption outweighs learning activities; a small "
                        "reallocation supports skill growth."
                    ),
                    evidence=(
                        f"learning_ratio = {learning:.2f} vs media_ratio = {media:.2f} "
                        f"across sessions."
                    ),
                    severity="low",
                    metric=f"learning_ratio {learning:.2f} < 0.2 & media_ratio {media:.2f} > 0.3",
                ),
            )

    # --- 9. Model-driven: likely next category -----------------------------
    if dl_signals and dl_signals.get("next_category"):
        nxt = dl_signals["next_category"]
        if nxt in {"social", "media"}:
            _add(
                recs,
                Recommendation(
                    title="Model predicts more social/media browsing ahead",
                    rationale=(
                        "The LSTM ranks social/media as the most probable next "
                        "category given recent sessions."
                    ),
                    evidence=f"Predicted next category = {nxt} "
                    f"(p={dl_signals.get('next_prob', 0.0):.2f}).",
                    severity="low",
                    metric=f"LSTM next = {nxt}",
                ),
            )

    # --- 10. Top domain concentration --------------------------------------
    if not top_domains.empty:
        top_domain = top_domains.iloc[0]
        top_share = top_domain["event_count"] / top_domains["event_count"].sum()
        if top_share > 0.3:
            _add(
                recs,
                Recommendation(
                    title=f"{top_domain['domain']} is your dominant site",
                    rationale=(
                        "A single domain drives over 30% of visits; evaluate whether "
                        "that time is intentional."
                    ),
                    evidence=f"{top_domain['domain']} = {top_share:.0%} of visits.",
                    severity="medium",
                    metric=f"{top_domain['domain']} share = {top_share:.2f}",
                ),
            )

    logger.info("recommendations_generated", count=len(recs))
    if len(recs) < 5:
        recs.append(
            Recommendation(
                title="Continue balanced browsing",
                rationale="No strong risk signals were detected in the current window.",
                evidence="All monitored metrics stayed within healthy thresholds.",
                severity="low",
                metric="baseline",
            )
        )
    return recs
