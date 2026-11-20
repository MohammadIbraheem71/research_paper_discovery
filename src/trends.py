"""Topic-share trend analysis.

Owner: Ibraheem. All stubs read the small tracked dummy CSV instead of real
topic assignments so they run offline in milliseconds.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.contracts import TrendData, TrendPoint
from src.settings_p2 import TOPIC_SHARE_DUMMY_PATH

logger = logging.getLogger(__name__)


def _load_dummy_share() -> pd.DataFrame:
    """Load the clearly-labelled dummy 5-topics x 8-quarters share table."""
    return pd.read_csv(TOPIC_SHARE_DUMMY_PATH)


def compute_topic_share(doc_topics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute each topic's share of papers per quarter.

    Purpose:
        Build the base table for every trend curve.
    Args:
        doc_topics_df: Paper-level topic assignments with quarter and label.
    Returns:
        DataFrame with columns ``[quarter, topic_id, label, share, count]``, e.g.
        ``pd.DataFrame([{"quarter": "2024Q1", "topic_id": 0, "label": "Topic 0",
        "share": 0.25, "count": 250}])``.
    Owner:
        Ibraheem.
    """
    logger.warning("STUB: compute_topic_share")
    # TODO(Ibraheem): group by (quarter, topic_id), count papers, divide by the
    # quarterly total and join the topic labels. Until then return dummy rows.
    return _load_dummy_share()


def fit_slopes(share_df: pd.DataFrame) -> pd.DataFrame:
    """Fit a linear trend to each topic's quarterly share.

    Purpose:
        Quantify rising vs declining topics with an OLS slope and p-value.
    Args:
        share_df: Output of :func:`compute_topic_share`.
    Returns:
        DataFrame with columns ``[topic_id, label, slope, p_value]``, e.g.
        ``pd.DataFrame([{"topic_id": 0, "label": "Topic 0", "slope": 0.01,
        "p_value": 0.02}])``.
    Owner:
        Ibraheem.
    """
    logger.warning("STUB: fit_slopes")
    # TODO(Ibraheem): regress share on a numeric quarter index per topic using
    # scipy.stats.linregress; keep only topics with enough quarters.
    return pd.DataFrame(
        [
            {"topic_id": i, "label": f"Topic {i}", "slope": 0.01 * (2 - i), "p_value": 0.01 * (i + 1)}
            for i in range(5)
        ]
    )


def rising_declining(slopes: pd.DataFrame, n: int = 5) -> tuple[list, list]:
    """Split topics into the n fastest rising and n fastest declining.

    Purpose:
        Shortlist topics for a natural-language trend answer.
    Args:
        slopes: Output of :func:`fit_slopes`.
        n: Number of topics per side.
    Returns:
        tuple[list, list], e.g. ``([0, 1], [4, 3])`` of topic ids.
    Owner:
        Ibraheem.
    """
    logger.warning("STUB: rising_declining")
    # TODO(Ibraheem): sort by slope and take the top n and bottom n ids.
    ordered = slopes.sort_values("slope", ascending=False)["topic_id"].tolist()
    return ordered[:n], ordered[-n:][::-1]


def trend_lookup(question: str, share_df: pd.DataFrame) -> TrendData:
    """Find the topic most relevant to a question and return its trend payload.

    Purpose:
        Bridge a natural-language question to the TrendData contract.
    Args:
        question: User question, e.g. "which topics are growing?".
        share_df: Output of :func:`compute_topic_share`.
    Returns:
        TrendData, e.g. ``{"topic_id": 0, "label": "Topic 0", "series":
        [{"quarter": "2024Q1", "share": 0.2, "count": 200}, ...], ...}``.
    Owner:
        Ibraheem.
    """
    logger.warning("STUB: trend_lookup")
    # TODO(Ibraheem): match topic labels in the question, otherwise pick the
    # strongest slope from fit_slopes(); assemble a real TrendData.
    rows = share_df[share_df["topic_id"] == 0]
    series: list[TrendPoint] = [
        TrendPoint(quarter=str(r.quarter), share=float(r.share), count=int(r.count))
        for r in rows.itertuples()
    ]
    return TrendData(
        topic_id=0,
        label="Topic 0",
        start_quarter=series[0]["quarter"],
        end_quarter=series[-1]["quarter"],
        series=series,
        slope=0.01,
        p_value=0.01,
        chart_path=None,
    )


def plot_topic_trend(trend_data: TrendData) -> str:
    """Plot a topic's share over time and return the output path.

    Purpose:
        Produce the chart referenced by ``TrendData.chart_path``.
    Args:
        trend_data: Data returned by :func:`trend_lookup`.
    Returns:
        str, e.g. ``"results/topic_trend_dummy.png"`` (no file is written by the stub).
    Owner:
        Ibraheem.
    """
    logger.warning("STUB: plot_topic_trend")
    # TODO(Ibraheem): plot the series to results/topic_trend.png under a results/
    # subpath that is gitignored, and return the actual path written.
    return str(Path("results") / "topic_trend_dummy.png")
