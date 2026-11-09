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
