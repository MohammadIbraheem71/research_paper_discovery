# Research Paper Discovery and Trend Analysis

Research paper discovery project for a data science elective at FAST NUCES.

Dataset: [arXiv Papers Cleaned on Kaggle](https://www.kaggle.com/datasets/mohammadibraheem71/arxiv-papers-cleaned/data)
(the dataset also contains the raw data).

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

### Sohaib write below

### Aliyan

- **Problem statement:** Build a reproducible pipeline that prepares arXiv papers for
  category classification and research-paper discovery. The pipeline should preserve
  the agreed dataset contract, avoid train/test leakage, and produce comparable text
  and numeric feature representations.
- **Preprocessing:** `notebooks/03_preprocessing.ipynb` loads the cleaned parquet
  dataset, validates the target and required fields, creates a stratified
  train/test split, and fits transformations on the training data only.
- **Feature preparation:** The notebook compares TF-IDF word n-grams (unigrams and
  bigrams, L2-normalized), sentence-transformer embeddings
  (`all-MiniLM-L6-v2`, generated for a reproducible 10,000-paper stratified
  subset to keep CPU execution practical), and scaled numeric features derived
  from paper metadata. TF-IDF and numeric features use the complete split.
- **How to run the notebooks:** Install the packages in `requirements.txt`, place
  `papers.parquet` at `data/papers.parquet` (or update the input path in the
  notebook), and run notebooks 01 through 03 in order. The preprocessing notebook
  writes its matrices and split metadata to `results/`.

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
