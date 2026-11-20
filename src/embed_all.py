"""Full-corpus sentence embeddings for papers.parquet (Kaggle / GPU only).

Run on Kaggle with the ``arxiv-papers-cleaned`` dataset attached and a GPU
enabled, for example::

    python src/embed_all.py \
        --papers /kaggle/input/arxiv-papers-cleaned/papers.parquet \
        --out /kaggle/working/embeddings.npy \
        --check results/embeddings_check.json

The script encodes ``text_raw`` with all-MiniLM-L6-v2, L2-normalizes every
vector, and saves ``embeddings.npy`` in the exact row order of papers.parquet.
It also writes a small JSON check (row count and SHA-256 of the id column) so
the alignment can be verified later. Do not run this on CPU or locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# When launched as `python src/embed_all.py`, the script directory (not the repo
# root) is first on sys.path, so add the repo root before importing project code.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.config import EMBEDDING_DIM, EMBEDDING_MODEL


def find_papers(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = [
        Path("/kaggle/input/arxiv-papers-cleaned/papers.parquet"),
        Path("data/papers.parquet"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find papers.parquet. Attach the arxiv-papers-cleaned "
        "Kaggle dataset or pass --papers."
    )


def id_sha256(ids: pd.Series) -> str:
    payload = "\n".join(ids.astype(str).tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", default=None, help="path to papers.parquet")
    parser.add_argument("--out", default="embeddings.npy", help="output .npy file path")
    parser.add_argument(
        "--check", default="results/embeddings_check.json",
        help="output JSON check path",
    )
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    papers_path = find_papers(args.papers)
    df = pd.read_parquet(papers_path, columns=["id", "text_raw"])
    if df["id"].isna().any() or df["text_raw"].isna().any():
        raise ValueError("id and text_raw must not contain nulls")
    if df["id"].duplicated().any():
        raise ValueError("id column must be unique")

    print(f"Embedding {len(df):,} rows from {papers_path} with {args.model}")
    encoder = SentenceTransformer(args.model)
    embeddings = encoder.encode(
        df["text_raw"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    assert embeddings.shape == (len(df), args.dim), (
        f"expected shape {(len(df), args.dim)}, got {embeddings.shape}"
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "embeddings are not L2-normalized"

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)

    check_path = Path(args.check).expanduser()
    check_path.parent.mkdir(parents=True, exist_ok=True)
    check = {
        "rows": int(len(df)),
        "embedding_dim": int(args.dim),
        "embedding_file": out_path.name,
        "id_sha256": id_sha256(df["id"]),
        "model": args.model,
        "batch_size": int(args.batch_size),
        "source": papers_path.name,
    }
    check_path.write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {embeddings.shape} embeddings to {out_path.resolve()}")
    print(f"Saved alignment check to {check_path.resolve()}")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
