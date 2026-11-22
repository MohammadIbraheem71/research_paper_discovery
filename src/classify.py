"""Category classification: TF-IDF baselines and an embedding classifier.

Owner: Sohaib. Real implementations replace the Phase 2 skeleton stubs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import CATEGORIES

logger = logging.getLogger(__name__)


def load_split() -> tuple[list[str], list[str]]:
    """Load the train/test paper ids from the Phase 1 split.

    Reads ``results/split_metadata.json`` and returns the two id lists in the
    order they were written (stable, so every experiment shares the split).
    """
    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "split_metadata.json"
    with path.open(encoding="utf-8") as fh:
        meta = json.load(fh)
    return list(meta["train_ids"]), list(meta["test_ids"])


def load_xy(df: pd.DataFrame, ids: list[str]) -> tuple[list[str], np.ndarray]:
    """Select rows by id and split into texts and encoded labels."""
    label_path = Path(__file__).resolve().parents[1] / "results" / "label_classes.json"
    with label_path.open(encoding="utf-8") as fh:
        classes = json.load(fh)["classes_"]
    class_to_idx = {name: i for i, name in enumerate(classes)}

    id_col = "id" if "id" in df.columns else df.columns[0]
    indexed = df.set_index(id_col)
    available = [i for i in ids if i in indexed.index]
    if len(available) != len(ids):
        missing = [i for i in ids if i not in indexed.index]
        logger.warning("load_xy: %d ids missing from df", len(missing))
    subset = indexed.loc[available]

    texts = subset["text_lemma"].astype(str).tolist()
    labels = subset["primary_category"].astype(str).map(
        lambda c: class_to_idx.get(c, -1)
    ).to_numpy(dtype=np.int64)
    return texts, labels


def build_tfidf(train_texts: list[str]) -> Any:
    """Fit a TF-IDF vectorizer on the training texts."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(train_texts)
    return vectorizer


def train_model(
    name: Literal["logreg", "nb", "svm", "logreg_emb"],
    X_train: Any,
    y_train: np.ndarray,
) -> Any:
    """Train one classifier variant."""
    if name == "logreg":
        model = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, random_state=42)
    elif name == "nb":
        model = MultinomialNB(alpha=1.0)
    elif name == "svm":
        model = LinearSVC(C=1.0, random_state=42)
    elif name == "logreg_emb":
        model = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"Unknown model name: {name}")
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """Score a fitted model and return a serializable metrics dictionary."""
    y_pred = model.predict(X_test)
    names = list(class_names) if class_names else list(CATEGORIES)
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(names))), zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(names))))
    per_class = {
        names[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
        }
        for i in range(len(names))
    }
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": cm.tolist(),
    }


def save_metrics(results: dict, path: str | Path) -> None:
    """Write a metrics dictionary to JSON or CSV, depending on the suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        rows = []
        models = results.get("models", results if isinstance(results, dict) else None)
        if isinstance(models, list):
            for m in models:
                row = {"model": m["model"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
                for cls, vals in m.get("per_class", {}).items():
                    for metric, value in vals.items():
                        row[f"{cls}_{metric}"] = value
                rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        return None
    with path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return None


def plot_confusion(cm: list[list[int]], class_names: list[str], path: str | Path) -> str:
    """Plot a confusion matrix and return the saved image path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(class_names)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=names, yticklabels=names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return str(path)
