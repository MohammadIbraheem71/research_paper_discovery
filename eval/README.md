# Evaluation file formats

Both files are tracked and are **hand-labelled**. Do not guess labels from a
model's output.

## `eval_routing.json`

A JSON list of objects with exactly two fields:

| Field | Type | Meaning |
|---|---|---|
| `question` | str | The question shown to the router. |
| `route` | str | One of `literature`, `trend`, `hybrid`, `fallback`. |

There are 8 examples: two per route. This file is the gold set for
`src.evaluate.router_accuracy` and for the route assertion in
`tests/test_skeleton.py`.

## `eval_retrieval.json`

A JSON list of objects with these fields:

| Field | Type | Meaning |
|---|---|---|
| `question` | str | The retrieval question. |
| `filters` | object | Optional `year_min`, `year_max`, `category`, `topic`. |
| `relevant_ids` | list[str] | arXiv ids judged relevant. **Currently empty on purpose.** |
| `needs_human_verification` | bool | `true` until a human fills `relevant_ids`. |

There are 3 examples. `relevant_ids` must be filled by hand after reading the
papers; `hit_rate_at_k` and `mrr` will return 0 on these rows until then. That
is expected for the skeleton and is not a failing test.
