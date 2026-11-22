# Figure index

Figures are produced by the cleaning notebook (`01_*`), the EDA notebook (`2_*`)
and the uncapped-counts notebook. Every figure is also displayed directly under
its code cell in the generating notebook.

| File | What it shows | Why it matters for modelling |
|---|---|---|
| `01_issues_summary.png` | Counts of each cleaning issue (missing, duplicates, withdrawn, short abstracts) | Shows which cleaning rules actually remove rows and why the row count drops from 120,000 to 119,919. |
| `01_length_outliers.png` | Abstract/title length distributions with IQR fences | Documents the length outliers that were kept rather than deleted; motivates truncation in embedding models. |
| `2_abstract_length_histogram.png` | Distribution of abstract word counts | The long right tail motivates a fixed token limit for text encoders. |
| `2_title_length_histogram.png` | Distribution of title word counts | Titles are short and concentrated, so they are cheap to include in text features. |
| `2_abstract_vs_title_boxplot.png` | Abstract versus title length side by side | Confirms abstracts carry most of the text and length alone is a weak signal. |
| `2_class_balance.png` | Class counts in the cleaned dataset | The 40k cap makes the sample nearly balanced, so accuracy is meaningful but macro-F1 is still reported. |
| `2_uncapped_class_balance.png` | Class counts before the cap, with the 40k line | Shows the real arXiv imbalance (cs.CV ~1.86x cs.CL) that the cap removes; any real-world claim must use these counts. |
| `2_year_category_trend.png` | Uncapped papers per year and category | Growth from 2018 to 2024 (about 3.9x) and the fastest growth in cs.CL; supports time-split robustness checks. |
| `2_time_trends_papers_per_year.png` | Papers per year in the cleaned sample | Shows how the cap changes the apparent category ranking over time; do not read it as real arXiv volume. |
| `2_time_trends_length_by_year.png` | Mean abstract/title length per year | Mild length drift over time is another reason to check performance by year. |
| `2_frequent_unigrams.png` | Most common words on `text_lemma` | Generic terms dominate overall frequency, so they must be down-weighted (TF-IDF) or excluded. |
| `2_frequent_bigrams.png` | Most common two-word phrases | Confirms that common phrases are mostly generic academic language. |
| `2_per_category_distinctive_unigrams.png` | Words most specific to each category | Strong category-specific vocabulary (NLP, vision, learning theory) supports a TF-IDF/linear baseline. |
| `2_per_category_distinctive_bigrams.png` | Phrases most specific to each category | Distinctive phrases reinforce that text is the main predictive signal. |
| `2_per_category_lengths.png` | Abstract/title length by category (boxplots) | Length differs only slightly between classes, so it is a weak standalone feature. |
| `2_per_category_abstract_histograms.png` | Full abstract-length distribution within each category | Shows skew and long tails per class; cs.CV is shifted slightly longer, which raises mild truncation risk. |
| `2_corr_heatmap.png` | Correlations among length, year and number of categories | All relationships are weak (|r| up to ~0.12), so the numeric features can be used together. |
| `2_monthly_volume.png` | Papers per month overall and per category | Reveals short-term seasonality and the steady growth trend hidden by yearly aggregation. |
| `2_quarterly_volume.png` | Papers per quarter overall and per category | A smoother view of growth and category ordering than the monthly chart. |
| `6_rising_topic_shares.png` | Quarterly share of the top 5 rising BERTopic topics | Shows which embedding-based themes gained share; slopes and p-values are in notebook 06. |
| `6_declining_topic_shares.png` | Quarterly share of the top 5 declining BERTopic topics | Shows the largest share losses, led by `Image / Segmentation / Detection`. |
| `6_topic_share_heatmap.png` | Heatmap of all 24 topics by 28 quarters | Makes topic coverage gaps and long-run share changes visible at a glance. |
