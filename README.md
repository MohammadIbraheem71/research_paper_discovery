# Research Paper Discovery and Trend Analysis

Research paper discovery project for a data science elective at FAST NUCES.

Dataset: [arXiv Papers Cleaned on Kaggle](https://www.kaggle.com/datasets/mohammadibraheem71/arxiv-papers-cleaned/data)
(the dataset also contains the raw data).

## Team contributions

### Ibraheem write below

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
