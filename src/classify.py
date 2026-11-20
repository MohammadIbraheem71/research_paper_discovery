"""Category classification skeleton (logreg / NB / SVM / embeddings).

Owner: Sohaib. Every function is an offline stub until the real pipeline lands.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import CATEGORIES

logger = logging.getLogger(__name__)


def load_split() -> tuple[list[str], list[str]]:
    """Load the train/test paper ids from the Phase 1 split.

    Purpose:
        Give every classification experiment the same reproducible split.
    Args:
        None.
    Returns:
        tuple[list[str], list[str]], e.g. ``(["id-1", ...], ["id-2", ...])``.
    Owner:
        Sohaib.
    """
    logger.warning("STUB: load_split")
    # TODO(Sohaib): read results/split_metadata.json (regenerate with notebook 03
    # if missing) and return the two id lists in a stable order.
    return ["train-0000", "train-0001"], ["test-0000", "test-0001"]


def load_xy(df: pd.DataFrame, ids: list[str]) -> tuple[list[str], np.ndarray]:
    """Select rows by id and split into texts and encoded labels.

    Purpose:
        Turn the cleaned dataframe into model inputs.
    Args:
        df: Cleaned papers dataframe.
        ids: Paper ids for this split.
    Returns:
        tuple[list[str], np.ndarray], e.g. ``(["abstract text"], array([0, 1]))``.
    Owner:
        Sohaib.
    """
    logger.warning("STUB: load_xy")
    # TODO(Sohaib): filter df to ids, use text_lemma as X and primary_category
    # encoded with results/label_classes.json as y.
    subset = df[df["id"].isin(ids)] if "id" in df.columns else df.iloc[:0]
    texts = subset.get("text_lemma", pd.Series(dtype=str)).astype(str).tolist()
    return texts, np.zeros(len(texts), dtype=np.int64)


def build_tfidf(train_texts: list[str]) -> Any:
    """Fit a TF-IDF vectorizer on the training texts.

    Purpose:
        Produce the sparse text features for the linear baselines.
    Args:
        train_texts: Lemmatized training documents.
    Returns:
        Any, e.g. a fitted ``TfidfVectorizer`` (stub returns ``None``).
    Owner:
        Sohaib.
    """
    logger.warning("STUB: build_tfidf")
    # TODO(Sohaib): TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True,
    # norm="l2") to match notebook 03; return the fitted vectorizer.
    return None


def train_model(
    name: Literal["logreg", "nb", "svm", "logreg_emb"],
    X_train: Any,
    y_train: np.ndarray,
) -> Any:
    """Train one classifier variant.

    Purpose:
        Single factory for the four comparison models.
    Args:
        name: One of ``"logreg"``, ``"nb"``, ``"svm"``, ``"logreg_emb"``.
        X_train: Training features (sparse TF-IDF or dense embeddings).
        y_train: Encoded training labels.
    Returns:
        Any, e.g. a fitted estimator (stub returns ``None``).
    Owner:
        Sohaib.
    """
    logger.warning("STUB: train_model")
    # TODO(Sohaib): dispatch to LogisticRegression / MultinomialNB / LinearSVC;
    # logreg_emb uses the same logistic regression on dense embeddings.
    return None


def evaluate(
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """Score a fitted model and return a serializable metrics dictionary.

    Purpose:
        One comparable metrics payload per model variant.
    Args:
        model: Fitted estimator.
        X_test: Test features.
        y_test: Encoded test labels.
        class_names: Label order from results/label_classes.json.
    Returns:
        dict, e.g. ``{"accuracy": 0.0, "macro_f1": 0.0, "per_class":
        {"cs.LG": {"precision": 0.0, "recall": 0.0, "f1": 0.0}}, "confusion":
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}``.
    Owner:
        Sohaib.
    """
    logger.warning("STUB: evaluate")
    # TODO(Sohaib): predict, then compute accuracy, macro F1, per-class precision/
    # recall/F1 and the confusion matrix as plain Python lists.
    names = class_names or CATEGORIES
    return {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "per_class": {n: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for n in names},
        "confusion": [[0 for _ in names] for _ in names],
    }


def save_metrics(results: dict, path: str | Path) -> None:
    """Write a metrics dictionary to JSON.

    Purpose:
        Persist classification results for the README and comparison table.
    Args:
        results: Output of :func:`evaluate`.
        path: Destination JSON path.
    Returns:
        None, e.g. ``None`` after writing the file.
    Owner:
        Sohaib.
    """
    logger.warning("STUB: save_metrics")
    # TODO(Sohaib): json.dump(results, ...) with parent.mkdir(parents=True).
    return None


def plot_confusion(cm: list[list[int]], class_names: list[str], path: str | Path) -> str:
    """Plot a confusion matrix and return the saved image path.

    Purpose:
        Visual comparison of per-class errors.
    Args:
        cm: Confusion matrix as nested lists.
        class_names: Label order.
        path: Destination PNG path.
    Returns:
        str, e.g. ``"results/confusion.png"`` (no file is written by the stub).
    Owner:
        Sohaib.
    """
    logger.warning("STUB: plot_confusion")
    # TODO(Sohaib): seaborn heatmap with annotations, tight layout, save to path.
    return str(path)
