# Research Paper Discovery and Trend Analysis

Research paper discovery project for a data science elective at FAST NUCES.

## Dataset

**Source.** The raw data is the
[arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
published by Cornell University on Kaggle. It is a JSON-lines file with one
paper per line (`id`, `title`, `abstract`, `categories`, version dates, and
more). The project streams it line by line instead of loading the whole
snapshot into memory.

**Processed data.** The Phase 1 files are published as a public Kaggle
Dataset: [arXiv Papers Cleaned](https://www.kaggle.com/datasets/mohammadibraheem71/arxiv-papers-cleaned).
It can be downloaded without a Kaggle login and contains:

| File | Rows | Contents |
|---|---:|---|
| `papers_raw.parquet` | 120,000 | Output of step A1, before cleaning |
| `papers.parquet` | 119,919 | Final cleaned dataset used by every notebook |
| `papers_sample.parquet` | 10,000 | Reproducible random sample (seed 42) for quick tests |

**Scope.** Primary categories `cs.LG`, `cs.CL` and `cs.CV`; first-version
submission date from 2018 to 2024 inclusive. The sample is capped at 40,000
papers per category, so it is balanced by design rather than a natural arXiv
sample. The pre-cap counts can be reproduced with
`src/uncapped_counts.py`; they explain why the sample was balanced and are
reported alongside the EDA class-balance section.

**Features.**

| Column | Type | Description |
|---|---|---|
| `id` | string | arXiv identifier |
| `title` | string | Paper title, lightly cleaned |
| `abstract` | string | Abstract, lightly cleaned |
| `text_raw` | string | Title + abstract with LaTeX/URLs removed; case and punctuation kept (embeddings, LLM, retrieval) |
| `text_lemma` | string | Lowercased, stop-word-free, lemmatized text (TF-IDF, LDA) |
| `primary_category` | string | Target label: `cs.LG`, `cs.CL` or `cs.CV` |
| `categories` | string | All arXiv categories, space-separated |
| `date` | datetime | First-version submission date |
| `year` | int | Submission year (2018–2024) |
| `month` | int | Submission month (1–12) |
| `quarter` | string | Submission quarter, e.g. `2021Q3` |

**Target variable.** `primary_category` is a three-class label. After cleaning
the classes are `cs.CL` 39,980, `cs.CV` 39,973 and `cs.LG` 39,966. Because the
40,000-per-category cap balances them, accuracy is meaningful but macro-F1 is
still reported; the balance is a sampling choice and not the real arXiv
distribution.

## Cleaning summary

Step A1 (`src/preprocessing.py`) filters the raw snapshot and step A2
(`src/cleaning.py`, documented in `notebooks/01_data_cleaning.ipynb`) cleans
the text and builds the final table.

| Step | Rows | What happened |
|---|---:|---|
| Raw filtered data (A1) | 120,000 | Kept only `cs.LG`, `cs.CL`, `cs.CV`, first-version years 2018–2024, capped at 40,000 per category (seed 42) |
| After missing / duplicates / withdrawn | 119,970 | Dropped 0 missing values, 0 duplicate ids, 29 near-identical titles and 1 withdrawn-paper notice |
| After abstracts under 30 words | 119,919 | Dropped 51 too-short abstracts (52 were flagged; one had already been removed) |

Other cleaning decisions:

- **Inconsistent text formats:** LaTeX formulas and commands, URLs, stray
  symbols, newlines and repeated spaces are removed; capitalisation and
  punctuation are kept in `text_raw`.
- **Two text versions:** `text_raw` is for sentence embeddings, the LLM and
  retrieval; `text_lemma` (lowercased, stop words and generic academic words
  removed, spaCy-lemmatized) is for TF-IDF and LDA.
- **Dates and categories:** `date` is the first-version submission date;
  `year`, `month` and `quarter` are derived from it. `primary_category` is the
  first listed arXiv category.
- **Outliers:** abstract/title lengths outside the IQR fences are documented
  but **kept**. They are legitimate papers, and deleting rows would break the
  row alignment with the shared sentence embeddings. The embedding model
  truncates very long abstracts, which is acceptable because the key content
  comes first.
- **Row order:** the final table is sorted once by `id` and must never be
  re-sorted, because the embeddings and all downstream results follow this
  order.
- **Validation:** no nulls, unique ids, sorted ids, expected categories, years
  2018–2024 and a valid `quarter` format are all checked before saving.

## Team contributions

### Ibraheem

- **Shared configuration:** `src/config.py` defines the agreed categories
  (`cs.LG`, `cs.CL`, `cs.CV`), the 2018–2024 date window, the per-category cap,
  the embedding model and dimension, and the global seed, so every step of the
  pipeline is reproducible.
- **Data acquisition and filtering (A1):** `src/preprocessing.py` streams the
  raw arXiv JSON snapshot line by line (memory-safe), keeps only papers whose
  primary category is in the agreed list and whose first-version date falls in
  the window, caps papers per category for balance, and writes
  `papers_raw.parquet` plus a reproducible 10,000-paper sample.
- **Cleaning pipeline (A2):** `src/cleaning.py` drops missing values, duplicate
  ids and near-identical normalised titles, removes withdrawn-paper notices and
  abstracts under 30 words, strips LaTeX/URLs/stray symbols, and builds the two
  text versions — `text_raw` (lightly cleaned, case and punctuation kept) and
  `text_lemma` (lowercased, stop words and generic academic words removed,
  spaCy-lemmatized). It derives `year`, `month` and `quarter`, enforces the
  shared column contract, and sorts once by `id` so all downstream steps stay
  aligned. Outputs `papers.parquet` and `papers_sample.parquet`.
- **Data cleaning notebook:** `notebooks/01_data_cleaning.ipynb` documents the
  pipeline end to end — missing values, duplicates, inconsistent formats
  (LaTeX, URLs, dates, categories), bad records, IQR length outliers, an issue
  summary, before/after examples, the two text versions, validation checks and
  saving. It produces `images/01_issues_summary.png` and
  `images/01_length_outliers.png`.
- **Topic modeling (A3):** `src/topics.py` builds a Gensim dictionary and
  bag-of-words corpus, tunes the number of LDA topics with `c_v` coherence
  (`tune`), trains the final multi-core LDA model, exports the top words per
  topic, and assigns a dominant topic and probability to every paper (`final`).
- **Dataset:** published the cleaned arXiv dataset on Kaggle
  (`mohammadibraheem71/arxiv-papers-cleaned`) and linked it in the README.

### Sohaib 

- **Exploratory data analysis (EDA):** `notebooks/02_eda.ipynb` loads the cleaned
  dataset (`papers.parquet`, 119,919 papers across `cs.LG`, `cs.CL` and `cs.CV`,
  2018 to 2024) and looks at it before modelling. It covers the basic structure,
  missing values, duplicates, text length, class balance, category overlap,
  vocabulary and feature correlations. Each plot is saved to `images/` and also
  shows up under its code cell. Horizontal bars, integer bins and separate
  panels keep labels and scales readable. Every section ends with a short
  written insight, and the notebook closes with what the findings mean for
  modelling.
- **Text length analysis:** Histograms and boxplots of abstract and title word
  counts (`abstract_length_histogram.png`, `title_length_histogram.png`,
  `abstract_vs_title_boxplot.png`). The median abstract is about 168 words, with
  the middle half running from roughly 140 to 200. The median title is about 9
  words (7 to 11), so abstracts are around 18 times longer. Abstract length has
  outliers at both ends, so a fixed truncation length is reasonable for
  embedding models.
- **Class balance and category overlap:** Each of the three classes has about
  40,000 papers because of the per-category cap, so no imbalance handling is
  needed. Cross-listing is common, though: 69,412 papers (about 58%) list more
  than one category. The most frequent pairs are `cs.AI` + `cs.LG` (16,797),
  `cs.LG` + `stat.ML` (14,293) and `cs.AI` + `cs.CL` (13,009), and `cs.LG`
  appears in five of the top ten pairs. `primary_category` is a single label on
  a partly ambiguous problem, so errors should be checked per class.
- **Vocabulary analysis:** The top 20 unigrams and bigrams
  (`frequent_unigrams.png`, `frequent_bigrams.png`) are counted on `text_lemma`,
  so stop words and generic academic phrases do not crowd the charts and the
  topical vocabulary shows through. Bigrams are counted within each paper.
- **Feature correlations:** The correlation heatmap (`corr_heatmap.png`) covers
  abstract length, title length, number of categories and year. All
  correlations are weak (|r| up to about 0.12), so these features can be used
  together without multicollinearity problems.
- **Implications for modelling:** Report macro-F1 next to accuracy and check
  per-class errors, especially `cs.LG` against `cs.CV` and `cs.CL`. Use
  `text_lemma` with TF-IDF for bag-of-words methods and `text_raw` for
  embeddings. Embedding models that read only about 256 tokens will truncate
  long abstracts.

### Aliyan

- **Problem statement:** Build a reproducible pipeline that prepares arXiv papers for
  category classification and research-paper discovery. The pipeline preserves the
  agreed dataset contract, prevents train/test leakage, and produces comparable text
  and numeric feature representations.
- **Preprocessing and normalization:** `notebooks/03_preprocessing.ipynb` loads the
  cleaned parquet dataset, validates the target and required fields, creates a
  stratified train/test split, and fits every transformation on the training data
  only. Word features use TF-IDF with unigram and bigram ranges and L2
  normalization; numeric metadata features are scaled using training data
  statistics.
- **Feature preparation:** The notebook compares TF-IDF features, sentence-transformer
  embeddings (`all-MiniLM-L6-v2`), and scaled numeric features derived from paper
  metadata. Embeddings are generated for a reproducible 10,000-paper stratified
  subset to keep CPU execution practical, while TF-IDF and numeric features use the
  complete split.
- **Reproducibility and repository structure:** Shared settings such as categories,
  date range, embedding model, embedding dimension, sample size, and random seed
  are defined in `src/config.py`. Generated matrices and split metadata belong in
  `results/`, local parquet inputs belong in `data/`, reusable code belongs in `src/`,
  and notebooks document each stage of the workflow.
- **How to run the notebooks:** Install the packages in `requirements.txt`, place
  `papers.parquet` at `data/papers.parquet` (or update the input path in the
  preprocessing notebook), and run notebooks 01 through 03 in order. The
  preprocessing notebook writes its matrices and split metadata to `results/`.

## Repository structure

```text
data/          # local parquet inputs; ignored by git
images/        # plots used in the EDA notebook
notebooks/     # cleaning, EDA, and preprocessing notebooks
results/       # generated matrices and split metadata; ignored by git
src/           # reusable project modules
```

The EDA notebook contains descriptions for each plotted image; the images are in
the `images/` folder.
