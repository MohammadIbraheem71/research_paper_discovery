# Phase 2 owners

## Rules

1. Edit only your own functions and files. If you need a change elsewhere, ask
   Ibraheem first.
2. Never change a function signature or `src/contracts.py` without telling
   Ibraheem. Those interfaces are frozen so teammates can work in parallel.
3. Keep the stubs working until your real version replaces them. `pytest -q`
   must stay green on `main`.
4. Pull before push. Do not rewrite shared history.
5. Commit from your own account in small commits with clear messages.
6. Never commit `*.parquet`, `*.npy`, `.env`, `*.db`, index files, or anything
   over 5 MB.

## Function ownership

| Function / symbol | File | Owner |
|---|---|---|
| `Route`, `Doc`, `Filters`, `Citation`, `TrendPoint`, `TrendData`, `GraphState` | `src/contracts.py` | Ibraheem (frozen) |
| `MAX_RETRIES`, `TOP_K`, `EMBEDDING_DIM`, `LLM_MODEL`, all paths | `src/settings_p2.py` | Ibraheem |
| `_cache_key`, `_cache_path` | `src/llm.py` | Aliyan |
| `call_with_retry`, `llm_call`, `llm_json` | `src/llm.py` | Aliyan |
| `_dummy_doc` | `src/rag.py` | Aliyan |
| `Resources`, `load_resources`, `build_index`, `embed_query`, `retrieve`, `generate_answer` | `src/rag.py` | Aliyan |
| `load_tokens`, `build_corpus`, `train_lda`, `coherence_cv` (LDA CLI), `top_words_table`, `assign_topics`, `cmd_tune`, `cmd_final` | `src/topics.py` | Ibraheem |
| `run_lda`, `run_bertopic`, `coherence_cv` (Phase 2 interface), `label_topics` | `src/topics.py` | Ibraheem |
| `_load_dummy_share` | `src/trends.py` | Ibraheem |
| `compute_topic_share`, `fit_slopes`, `rising_declining`, `trend_lookup`, `plot_topic_trend` | `src/trends.py` | Ibraheem |
| `load_split`, `load_xy`, `build_tfidf`, `train_model`, `evaluate`, `save_metrics`, `plot_confusion` | `src/classify.py` | Sohaib |
| `_calls`, `_extract_filters` | `src/nodes.py` | Sohaib |
| `router_node`, `fallback_node`, `synthesize_node` | `src/nodes.py` | Sohaib |
| `query_rewrite_node`, `retrieve_node`, `grade_docs_node`, `generate_answer_node` | `src/nodes.py` | Aliyan |
| `trend_lookup_node` | `src/nodes.py` | Ibraheem |
| `_route_edge`, `_grade_edge`, `_fanout` | `src/graph.py` | Ibraheem |
| `get_checkpointer`, `build_graph`, `run_question` | `src/graph.py` | Ibraheem |
| `hit_rate_at_k`, `mrr`, `run_ablation`, `citation_sample_csv` | `src/evaluate.py` | Aliyan |
| `router_accuracy` | `src/evaluate.py` | Sohaib |
| `question_id`, `load_eval_items`, `md_cell`, `build_pool` | `src/label_review.py` | Ibraheem |
| `main` (pooled label-review generator) | `src/build_label_review.py` | Ibraheem |
| `main` (human label application) | `src/apply_labels.py` | Ibraheem |
| Streamlit shell | `app.py` | Aliyan |

## File ownership

| File / directory | Owner |
|---|---|
| `notebooks/05_topic_modeling.ipynb`, `06_trends.ipynb`, `09_langgraph.ipynb` | Ibraheem |
| `notebooks/07_classification.ipynb` | Sohaib |
| `notebooks/08_rag.ipynb`, `10_evaluation.ipynb` (retrieval/ablation sections) | Aliyan |
| `notebooks/10_evaluation.ipynb` (router section) | Sohaib |
| `notebooks/00_kaggle_embeddings.ipynb` | Ibraheem (Phase 1) |
| `eval/eval_routing.json` | Sohaib (hand labels) |
| `eval/eval_retrieval.json` | Aliyan (hand labels) |
| `results/label_review_pooled.md`, `eval/label_decisions.csv` | Ibraheem (generated; human-filled) |
| `eval/README.md`, `OWNERS.md`, `README.md`, `tests/test_skeleton.py` | Ibraheem |
| `src/label_review.py`, `src/build_label_review.py`, `src/apply_labels.py` | Ibraheem |

`src/apply_labels.py` mechanically copies human decisions into
`eval/eval_retrieval.json`. It never decides relevance and never edits a label
itself.
