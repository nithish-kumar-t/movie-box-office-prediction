# README: Movie Box Office Prediction

## Project Overview
This project focuses on predicting movie box office revenue using structured metadata and enriched social sentiment/emotion signals. The goal is to understand how public perception influences movie performance and improve prediction accuracy.

## Dataset
We used an enriched version of the TMDB dataset, which includes the following key features:
- Basic metadata: budget, runtime, vote average/count, release date, etc.
- Categorical: language, status, director, production company, genres
- Engineered: release year/month, cast/genre/keyword counts
- Social Signals (new): sentiment and emotion scores

## Workflow Summary

### Step 1: Data Cleaning & Preprocessing
- Removed missing/invalid rows
- Normalized column names
- Converted `release_date` to datetime
- Added features like `release_year`, `release_month`, `cast_count`, `genre_count`, `keyword_count`
- Extracted top 50 directors and production companies; encoded others as 'other'
- One-hot encoded categorical columns for XGBoost and LightGBM
- For CatBoost, used categorical column support directly

### Step 2: Baseline Modeling (Before Sentiment)
We trained 3 models using only structured features (metadata):

#### Model 1: XGBoost
- Validation Accuracy (Custom ±10M): 65.23%
- Test Accuracy (Custom ±10M): 64.78%

#### Model 2: CatBoost
- Validation Accuracy: 65.05%
- Test Accuracy: 64.94%

#### Model 3: LightGBM
- Validation Accuracy: 65.42%
- Test Accuracy: 65.68%

---

### Step 3: Sentiment & Emotion Signal Integration
We extracted public sentiment and emotional reactions using ML models. Due to computational constraints, instead of fetching data for all movies from Reddit and YouTube APIs, we generated sentiment and emotion scores using pretrained models.

#### Added 9 new features:
- Sentiment: `very_negative`, `neutral`, `very_positive`
- Emotion: `anger`, `disgust`, `joy`, `sadness`, etc.
- These columns were given 3x weight during model training

### Step 4: Model Re-training (After Sentiment Enrichment)

#### Model 1: XGBoost
- Validation Accuracy: 84.86%
- Test Accuracy: 85.12%

#### Model 2: CatBoost
- Validation Accuracy: 88.54%
- Test Accuracy: 89.30%

#### Model 3: LightGBM
- Validation Accuracy: 78.63%
- Test Accuracy: 78.71%

---

## Key Insights
- Sentiment and emotion scores significantly improved prediction accuracy.
- CatBoost emerged as the top performer after enrichment.
- Weighting social features helped the model better understand audience anticipation and emotion.

## Future Work
- Automate real-time data fetching via Reddit/YouTube APIs
- Perform time-series analysis of sentiment evolution before and after release
- Try deep learning with attention over time-stamped comments
- Deploy as an interactive dashboard or web app

---

## Tech Stack
- Python, Pandas, NumPy
- XGBoost, CatBoost, LightGBM
- HuggingFace Transformers (DistilBERT)
- Text2Emotion for emotion scoring
- Matplotlib for visualization

---

## Output File
- Final enriched dataset: `cleaned_movies_with_sentiment.csv`

---

## Author
Niharika Belavadi Shekar


