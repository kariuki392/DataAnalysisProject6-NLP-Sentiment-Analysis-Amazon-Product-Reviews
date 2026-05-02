# NLP Sentiment Analysis — Amazon Product Reviews

A Jupyter notebook that explores **Natural Language Processing** on consumer electronics reviews: sentiment labeling with TextBlob, exploratory visualizations by product and time, TF‑IDF feature extraction, and supervised classifiers (logistic regression, random forest, gradient boosting) trained on rating-based labels.

## Contents

| Path | Description |
|------|-------------|
| `notebooks/nlp_sentiment_analysis.ipynb` | Main analysis notebook |
| `data/raw/7817_1.csv` | Raw review export (Amazon-style product metadata + review fields) |

The notebook summarizes roughly **1,597 reviews** across **62 products** (Amazon Devices / electronics focus).

## Requirements

- **Python** 3.9+ recommended  
- **Packages:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `textblob`, `wordcloud`, `scikit-learn`

Install dependencies (example):

```bash
pip install pandas numpy matplotlib seaborn textblob wordcloud scikit-learn
```

[TextBlob](https://textblob.readthedocs.io/) may prompt for NLTK corpora on first use; follow any on-screen download instructions if sentiment calls fail.

## Running the notebook

1. Clone or download this repository.
2. The first code cell loads the CSV with `pd.read_csv('7817_1.csv')`, so Jupyter’s **working directory** must contain that file **or** you should change the path in the notebook.

   From the repo root, a simple approach is to copy the dataset next to the notebook:

   ```bash
   cp data/raw/7817_1.csv notebooks/
   ```

   Then start Jupyter from `notebooks/` **or** open the notebook and set the kernel working directory appropriately.

   Alternatively, edit the load line to:

   ```python
   df = pd.read_csv('../data/raw/7817_1.csv')
   ```

   if you keep the CSV only under `data/raw/` and run with the notebook’s directory as `notebooks/`.

3. Launch Jupyter and run all cells:

   ```bash
   cd notebooks
   jupyter notebook nlp_sentiment_analysis.ipynb
   ```

## What the notebook covers

1. **Setup & data loading** — Column selection, cleaning, product name simplification  
2. **Text preprocessing** — Normalization for downstream NLP  
3. **Sentiment (TextBlob)** — Polarity, subjectivity, positive / neutral / negative labels  
4. **EDA** — Distributions, product-level sentiment, timelines, helpful votes  
5. **Modeling** — Train/test split, TF‑IDF pipelines, cross-validation, ROC and confusion matrices, interpretable coefficients / keyword highlights  
6. **Themes** — Bigrams, correlations, recommendation breakdowns  

## Dataset note

The file `7817_1.csv` is a labeled product-review export (IDs, brands, ratings, review text/titles, etc.). Use it in line with the original dataset’s license and terms.

## License

Add a license file here if you plan to distribute the project; the notebook and structure are yours to pair with whatever terms apply to the data you use.
