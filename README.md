# Sentiment Analysis: VADER vs Scikit-learn

A comparison of lexicon-based and machine learning approaches for sentiment analysis using Amazon product reviews.

## Overview

This project compares two different sentiment analysis methods:
- **VADER**: A lexicon-based approach using NLTK's VADER sentiment analyzer
- **Scikit-learn**: A machine learning approach using TF-IDF vectorization and LinearSVC

The analysis is performed on a dataset of 10,000 Amazon product reviews with pre-labeled positive and negative sentiments.

## Dataset

- **Source**: Amazon product reviews
- **Size**: 10,000 reviews
- **Format**: TSV file with 'label' and 'review' columns
- **Labels**: 'pos' (positive) and 'neg' (negative)

## Methods

### VADER Sentiment Analysis
- Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon
- Rule-based approach with predefined sentiment scores
- No training required
- Fast and lightweight

### Scikit-learn Pipeline
- TF-IDF vectorization for text feature extraction
- Linear Support Vector Classification (LinearSVC)
- Train/test split (67%/33%)
- Requires labeled training data

## Results

| Model | Accuracy | Negative Precision | Positive Precision | Negative Recall | Positive Recall |
|-------|----------|-------------------|-------------------|-----------------|-----------------|
| VADER | 0.71 | 0.86 | 0.64 | 0.52 | 0.91 |
| Scikit-learn | 0.87 | 0.86 | 0.89 | 0.89 | 0.85 |

## Key Findings

- Scikit-learn achieved higher overall accuracy (87% vs 71%)
- VADER shows bias toward positive predictions
- Scikit-learn provides more balanced performance across classes
- VADER struggles with negative sentiment detection (52% recall)

## Files

- `sentiment_analysis_comparison.ipynb`: Main analysis notebook
- `amazonreviews.tsv`: Dataset file
- `README.md`: This file

## Requirements

- Python 3.7+
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn

## Usage

1. Upload the notebook to Google Colab or run locally
2. Upload the `amazonreviews.tsv` file when prompted
3. Run all cells to see the complete analysis

## Notes

- The notebook is designed to run in Google Colab
- File upload is required for the dataset
- Results may vary slightly due to random train/test splitting
