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
    import json

    from src.settings_p2 import ROOT

    if "quarter" not in doc_topics_df or "bertopic_topic" not in doc_topics_df:
        raise ValueError("doc_topics_df must have 'quarter' and 'bertopic_topic' columns")
    if doc_topics_df.empty:
        raise ValueError("doc_topics_df is empty; real topic assignments are required")

    # Outlier topic -1 is excluded from the numerator: HDBSCAN could not assign
    # those papers, so they must not be counted as a topic. They DO remain in
    # the quarterly denominator, which is all papers in that quarter.
    assigned = doc_topics_df[doc_topics_df["bertopic_topic"] != -1]
    quarterly_total = doc_topics_df.groupby("quarter").size().rename("quarter_total")
    counts = (
        assigned.groupby(["quarter", "bertopic_topic"]).size().rename("count").reset_index()
        .rename(columns={"bertopic_topic": "topic_id"})
        .join(quarterly_total, on="quarter")
    )
    counts["share"] = counts["count"] / counts["quarter_total"]

    labels: dict[int, str] = {}
    if "label" in assigned:
        labels = dict(zip(assigned["topic_id"], assigned["label"])) if "topic_id" in assigned else {}
    if not labels:
        path = ROOT / "results" / "topics_bertopic.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            labels = {int(t["topic_id"]): t["label"] for t in payload["topics"]}
    counts["label"] = counts["topic_id"].map(labels).fillna(
        counts["topic_id"].map(lambda t: f"Topic {t}")
    )
    out = counts[["quarter", "topic_id", "label", "share", "count"]].sort_values(
        ["quarter", "topic_id"]
    ).reset_index(drop=True)
    out["topic_id"] = out["topic_id"].astype(int)
    out["count"] = out["count"].astype(int)
    return out


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
    from scipy.stats import linregress

    def quarter_index(q: str) -> int:
        year, quarter = str(q).split("Q")
        return int(year) * 4 + int(quarter) - 1

    rows = []
    for topic_id, group in share_df.groupby("topic_id"):
        g = group.sort_values("quarter")
        if len(g) < 3:
            continue
        x = [quarter_index(q) for q in g["quarter"]]
        fit = linregress(x, g["share"].astype(float).tolist())
        rows.append({
            "topic_id": int(topic_id),
            "label": str(g["label"].iloc[0]),
            "slope": float(fit.slope),
            "p_value": float(fit.pvalue),
        })
    return pd.DataFrame(rows, columns=["topic_id", "label", "slope", "p_value"])


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
    ordered = slopes.sort_values(
        ["slope", "topic_id"], ascending=[False, True]
    )["topic_id"].tolist()
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
    import re
    from difflib import SequenceMatcher

    slopes = fit_slopes(share_df)
    if slopes.empty:
        raise ValueError("No topic slopes available from share_df")

    # Fuzzy match: word overlap plus difflib.SequenceMatcher ratio against each
    # topic label. If no label matches at all, use the strongest rising slope.
    question_lower = question.lower()
    question_words = set(re.findall(r"[a-z]+", question_lower))
    best_score, chosen = -1.0, None
    for row in slopes.itertuples(index=False):
        label = str(row.label).lower()
        label_words = set(re.findall(r"[a-z]+", label))
        score = len(question_words & label_words) + SequenceMatcher(None, question_lower, label).ratio()
        if score > best_score:
            best_score, chosen = score, row
    if best_score <= 0:
        chosen = slopes.loc[slopes["slope"].idxmax()]

    rows = share_df[share_df["topic_id"] == int(chosen.topic_id)].sort_values("quarter")
    quarter_tags = re.findall(r"\b(20\d{2})Q([1-4])\b", question, flags=re.IGNORECASE)
    if not quarter_tags:
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]
    else:
        years = [int(y) for y, _ in quarter_tags]
    if years:
        lo, hi = min(years), max(years)
        filtered = rows[rows["quarter"].str[:4].astype(int).between(lo, hi)]
        if not filtered.empty:
            rows = filtered

    series: list[TrendPoint] = [
        TrendPoint(quarter=str(r.quarter), share=float(r.share), count=int(r.count))
        for r in rows.itertuples()
    ]
    if not series:
        raise ValueError("Time range in question produced an empty trend series")
    return TrendData(
        topic_id=int(chosen.topic_id),
        label=str(chosen.label),
        start_quarter=series[0]["quarter"],
        end_quarter=series[-1]["quarter"],
        series=series,
        slope=float(chosen.slope),
        p_value=float(chosen.p_value),
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.settings_p2 import ROOT

    out_dir = ROOT / "results" / "trends"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"topic_trend_{trend_data['topic_id']}.png"
    quarters = [point["quarter"] for point in trend_data["series"]]
    shares = [point["share"] for point in trend_data["series"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(quarters, shares, marker="o", linewidth=1.5)
    ax.set_title(f"{trend_data['label']} — quarterly share")
    ax.set_ylabel("share of papers")
    ax.set_xlabel("quarter")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=60)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
