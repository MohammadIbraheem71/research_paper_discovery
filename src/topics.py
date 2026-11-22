"""A3: LDA baseline with Gensim.

Two steps (run in order):
  python -m src.topics tune  --papers data/papers.parquet --out results   # pick k by c_v coherence
  python -m src.topics final --papers data/papers.parquet --out results --k 20   # train on all papers
"""
import logging
import os
import pickle
import multiprocessing as mp

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel

from src.config import SEED

logger = logging.getLogger(__name__)


def load_tokens(papers_path, n=None):
    df = pd.read_parquet(papers_path, columns=["id", "text_lemma"])
    if n and n < len(df):
        df = df.sample(n, random_state=SEED).sort_values("id")
    return df["id"].tolist(), [t.split() for t in df["text_lemma"]]


def build_corpus(tokens, no_below=20, no_above=0.4, keep_n=20000):
    """Dictionary + bag-of-words. Drops very rare words and words in >40% of papers."""
    dct = Dictionary(tokens)
    dct.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dct.doc2bow(t) for t in tokens]
    return dct, corpus


def train_lda(corpus, dct, k, passes=5, workers=None):
    workers = workers or max(1, mp.cpu_count() - 1)
    return LdaMulticore(corpus=corpus, id2word=dct, num_topics=k, passes=passes,
                        chunksize=4000, workers=workers, random_state=SEED,
                        eval_every=None)


def _coherence_cv_lda(model, tokens, dct, topn=10):
    cm = CoherenceModel(model=model, texts=tokens, dictionary=dct,
                        coherence="c_v", topn=topn, processes=max(1, mp.cpu_count() - 1))
    return cm.get_coherence()


def top_words_table(model, topn=15):
    rows = []
    for t in range(model.num_topics):
        words = [w for w, _ in model.show_topic(t, topn=topn)]
        rows.append({"lda_topic": t, "top_words": ", ".join(words), "topic_label": ""})
    return pd.DataFrame(rows)


def assign_topics(model, corpus, ids):
    """Dominant topic and its probability for every paper."""
    out = []
    for bow in corpus:
        dist = model.get_document_topics(bow, minimum_probability=0.0)
        t, p = max(dist, key=lambda x: x[1]) if dist else (-1, 0.0)
        out.append((int(t), float(p)))
    return pd.DataFrame({"id": ids, "lda_topic": [o[0] for o in out], "lda_prob": [o[1] for o in out]})


def cmd_tune(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    ids, tokens = load_tokens(a.papers, a.sample)
    dct, corpus = build_corpus(tokens)
    print(f"tuning on {len(tokens)} papers, vocabulary {len(dct)} words")
    rows = []
    for k in a.ks:
        m = train_lda(corpus, dct, k, passes=a.passes)
        c = coherence_cv(m, tokens, dct)
        rows.append({"k": k, "c_v": round(c, 4)})
        print(f"k={k:3d}  c_v={c:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/lda_coherence.csv", index=False)
    plt.figure(figsize=(6, 4)); plt.plot(res.k, res.c_v, marker="o")
    plt.xlabel("number of topics (k)"); plt.ylabel("c_v coherence"); plt.title("LDA coherence vs k")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{a.out}/lda_coherence.png", dpi=150)
    print("best k by c_v:", int(res.loc[res.c_v.idxmax(), "k"]))


def cmd_final(a):
    os.makedirs(a.out, exist_ok=True)
    ids, tokens = load_tokens(a.papers)
    dct, corpus = build_corpus(tokens)
    print(f"training k={a.k} on {len(tokens)} papers, vocabulary {len(dct)} words")
    model = train_lda(corpus, dct, a.k, passes=a.passes)
    model.save(f"{a.out}/lda_model")
    with open(f"{a.out}/lda_dictionary.pkl", "wb") as f:
        pickle.dump(dct, f)
    words = top_words_table(model)
    words.to_csv(f"{a.out}/lda_topic_words.csv", index=False)
    print(words[["lda_topic", "top_words"]].to_string(index=False))
    assigned = assign_topics(model, corpus, ids)
    assigned.to_parquet(f"{a.out}/lda_topics.parquet", index=False)
    print("\nc_v coherence (all papers):", round(coherence_cv(model, tokens, dct), 4))
    print("papers per topic:\n", assigned.lda_topic.value_counts().sort_index().to_string())
    print("saved to", a.out)



# --- Phase 2 stubs (owner: Ibraheem) -------------------------------------
# The LDA scaffold above is unchanged in behaviour. coherence_cv below is a
# compatibility dispatcher: the legacy (model, tokens, dct) call still works,
# while the new Phase 2 (topic_words, tokenized_texts) interface is available.


def run_lda(texts: list[str], k: int = 20, seed: int = 42) -> dict:
    """Train a k-topic LDA model and summarize it.

    Purpose:
        Phase 2 entry point for the LDA baseline on ``text_lemma`` documents.
    Args:
        texts: Tokenized documents as whitespace-separated strings.
        k: Number of topics.
        seed: Random seed, defaults to the project seed.
    Returns:
        dict, e.g. ``{"k": 20, "topic_words": [[...]], "model_path": "..."}``.
    Owner:
        Ibraheem.
    """
    from src.settings_p2 import CACHE_DIR

    tokens = [t.split() for t in texts]
    dct, corpus = build_corpus(tokens)
    # workers is a resource setting only: 4 keeps peak RSS bounded on the
    # 119,919-document corpus. Model parameters (k, passes, seed) are unchanged.
    model = train_lda(corpus, dct, k, passes=5, workers=4)
    topic_words = [
        [word for word, _ in model.show_topic(t, topn=10)] for t in range(k)
    ]
    doc_topics = []
    for bow in corpus:
        dist = model.get_document_topics(bow, minimum_probability=0.0)
        doc_topics.append(int(max(dist, key=lambda x: x[1])[0]) if dist else -1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CACHE_DIR / "lda_model"
    model.save(str(model_path))
    return {
        "k": k,
        "seed": seed,
        "topic_words": topic_words,
        "doc_topics": doc_topics,
        "model_path": str(model_path),
    }


def run_bertopic(texts: list[str], embeddings, nr_topics: int | str = "auto") -> dict:
    """Fit BERTopic on texts and precomputed embeddings.

    Purpose:
        Produce neural topics and per-document topic assignments.
    Args:
        texts: Raw abstracts/titles.
        embeddings: Array of shape ``(n, 384)`` from data/embeddings.npy.
        nr_topics: Target topic count or ``"auto"``.
    Returns:
        dict, e.g. ``{"nr_topics": 5, "topic_words": [[...]], "doc_topics": [...]}``.
    Owner:
        Ibraheem.
    """
    import hdbscan
    import numpy as np
    import umap
    from bertopic import BERTopic

    from src.settings_p2 import CACHE_DIR

    # UMAP/HDBSCAN settings are the standard BERTopic defaults; random_state
    # makes the projection reproducible and low_memory bounds peak RSS.
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        random_state=42,
        low_memory=True,
        n_jobs=4,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=50,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    topic_model = BERTopic(
        embedding_model=np.asarray(embeddings),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        random_state=42,
        calculate_probabilities=False,
        verbose=True,
    )
    doc_topics, _ = topic_model.fit_transform(list(texts), np.asarray(embeddings))
    doc_topics = [int(t) for t in doc_topics]

    topic_ids = [
        int(t) for t in topic_model.get_topic_info()["Topic"] if int(t) != -1
    ]
    topic_words = [
        [word for word, _ in topic_model.get_topic(t)[:10]] for t in topic_ids
    ]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(CACHE_DIR / "bertopic_model")
    topic_model.save(model_path, serialization="pickle")

    return {
        "nr_topics": len(topic_ids),
        "topic_ids": topic_ids,
        "topic_words": topic_words,
        "doc_topics": doc_topics,
        "outlier_share": float(np.mean(np.array(doc_topics) == -1)),
        "model_path": model_path,
    }


def coherence_cv(
    topic_words: list[list[str]] | object,
    tokenized_texts: list[list[str]],
    dct: object | None = None,
    topn: int = 10,
) -> float:
    """Compute c_v coherence, accepting both the new and legacy call shapes.

    Purpose:
        Score topic quality from top words, or from a fitted Gensim model when
        called by the existing LDA CLI: ``coherence_cv(model, tokens, dct)``.
    Args:
        topic_words: List of top-word lists, or a Gensim LDA model (legacy).
        tokenized_texts: Tokenized documents, or the legacy ``tokens`` argument.
        dct: Gensim dictionary; required for the legacy call shape.
        topn: Number of top words considered by the legacy Gensim computation.
    Returns:
        float, e.g. ``0.0`` from the stub; legacy calls return the real c_v value.
    Owner:
        Ibraheem.
    """
    if hasattr(topic_words, "show_topic"):  # legacy Gensim model call
        return _coherence_cv_lda(topic_words, tokenized_texts, dct, topn=topn)
    if dct is None:
        dct = Dictionary(tokenized_texts)
    # processes=1 avoids copying the 119,919-document token lists into every
    # worker; the score is identical for either model and both use the same
    # tokenized_texts, so the comparison is fair.
    cm = CoherenceModel(
        topics=list(topic_words),
        texts=tokenized_texts,
        dictionary=dct,
        coherence="c_v",
        topn=topn,
        processes=1,
    )
    return float(cm.get_coherence())


def label_topics(topic_words: list[list[str]]) -> dict[int, str]:
    """Assign a short human-readable label to each topic id.

    Purpose:
        Turn top words into chart-friendly topic labels.
    Args:
        topic_words: Top words per topic id.
    Returns:
        dict[int, str], e.g. ``{0: "Topic 0"}``.
    Owner:
        Ibraheem.
    """
    logger.warning("STUB: label_topics")
    # TODO(Ibraheem): use an LLM or a top-word heuristic to name each topic.
    return {i: f"Topic {i}" for i in range(len(topic_words))}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("tune")
    t.add_argument("--papers", required=True); t.add_argument("--out", default="results")
    t.add_argument("--ks", type=int, nargs="+", default=[10, 15, 20, 30])
    t.add_argument("--sample", type=int, default=20000, help="papers used for tuning")
    t.add_argument("--passes", type=int, default=5)
    t.set_defaults(fn=cmd_tune)
    f = sp.add_parser("final")
    f.add_argument("--papers", required=True); f.add_argument("--out", default="results")
    f.add_argument("--k", type=int, required=True); f.add_argument("--passes", type=int, default=8)
    f.set_defaults(fn=cmd_final)
    a = p.parse_args(); a.fn(a)
