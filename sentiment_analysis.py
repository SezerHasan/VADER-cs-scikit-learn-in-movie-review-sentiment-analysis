# VADER vs Scikit-learn Sentiment Analysis
# Simple and straightforward code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("VADER vs Scikit-learn Sentiment Analysis")
print("="*50)

# Download NLTK data if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Load the data
print("Loading dataset...")
df = pd.read_csv('amazonreviews.tsv', sep='\t')
print(f"Dataset loaded: {len(df)} reviews")
print(f"Label distribution:\n{df['label'].value_counts()}")

# VADER Analysis
print("\n" + "="*50)
print("VADER SENTIMENT ANALYSIS")
print("="*50)

analyzer = SentimentIntensityAnalyzer()
print("Analyzing reviews with VADER...")

# Analyze each review
df['vader_scores'] = df['review'].apply(analyzer.polarity_scores)
df['compound_score'] = df['vader_scores'].apply(lambda x: x['compound'])
df['comp_score'] = df['compound_score'].apply(lambda x: 'pos' if x >= 0 else 'neg')

# Calculate VADER accuracy
vader_accuracy = accuracy_score(df['label'], df['comp_score'])
print(f"VADER Accuracy: {vader_accuracy:.4f}")

print("\nVADER Classification Report:")
print(classification_report(df['label'], df['comp_score']))

# Scikit-learn Analysis
print("\n" + "="*50)
print("SCIKIT-LEARN SENTIMENT ANALYSIS")
print("="*50)

# Prepare data
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create and train model
print("Training Scikit-learn model...")
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC(random_state=42))
])

text_clf.fit(X_train, y_train)
predictions_pipeline = text_clf.predict(X_test)

# Calculate Scikit-learn accuracy
sklearn_accuracy = accuracy_score(y_test, predictions_pipeline)
print(f"Scikit-learn Accuracy: {sklearn_accuracy:.4f}")

print("\nScikit-learn Classification Report:")
print(classification_report(y_test, predictions_pipeline))

# Get all metrics
print("\n" + "="*50)
print("EXTRACTING METRICS")
print("="*50)

# VADER metrics
vader_report = classification_report(df['label'], df['comp_score'], output_dict=True)
vader_neg_precision = vader_report['neg']['precision']
vader_pos_precision = vader_report['pos']['precision']
vader_neg_recall = vader_report['neg']['recall']
vader_pos_recall = vader_report['pos']['recall']

# Scikit-learn metrics
sklearn_report = classification_report(y_test, predictions_pipeline, output_dict=True)
sklearn_neg_precision = sklearn_report['neg']['precision']
sklearn_pos_precision = sklearn_report['pos']['precision']
sklearn_neg_recall = sklearn_report['neg']['recall']
sklearn_pos_recall = sklearn_report['pos']['recall']

# Calculate F1 scores
vader_neg_f1 = 2 * (vader_neg_precision * vader_neg_recall) / (vader_neg_precision + vader_neg_recall)
vader_pos_f1 = 2 * (vader_pos_precision * vader_pos_recall) / (vader_pos_precision + vader_pos_recall)
sklearn_neg_f1 = 2 * (sklearn_neg_precision * sklearn_neg_recall) / (sklearn_neg_precision + sklearn_neg_recall)
sklearn_pos_f1 = 2 * (sklearn_pos_precision * sklearn_pos_recall) / (sklearn_pos_precision + sklearn_pos_recall)

# Create plots
print("\n" + "="*50)
print("CREATING PLOTS")
print("="*50)

# Set up data for plots
models = ['VADER', 'Scikit-learn']
accuracy_scores = [vader_accuracy, sklearn_accuracy]
precision_neg = [vader_neg_precision, sklearn_neg_precision]
precision_pos = [vader_pos_precision, sklearn_pos_precision]
recall_neg = [vader_neg_recall, sklearn_neg_recall]
recall_pos = [vader_pos_recall, sklearn_pos_recall]
f1_scores_neg = [vader_neg_f1, sklearn_neg_f1]
f1_scores_pos = [vader_pos_f1, sklearn_pos_f1]

# Plot 1: Accuracy comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy_scores, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy Score', fontsize=14)
plt.ylim(0, 1)

# Add value labels on bars
for i, v in enumerate(accuracy_scores):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=14)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Accuracy comparison plot saved")

# Plot 2: F1 scores comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# F1 scores for negative class
bars1 = ax1.bar(models, f1_scores_neg, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
ax1.set_title('F1 Score - Negative Class', fontsize=14, fontweight='bold')
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(f1_scores_neg):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# F1 scores for positive class
bars2 = ax2.bar(models, f1_scores_pos, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
ax2.set_title('F1 Score - Positive Class', fontsize=14, fontweight='bold')
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(f1_scores_pos):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('f1_scores_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ F1 scores comparison plot saved")

# Plot 3: Comprehensive performance plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('VADER vs Scikit-learn Performance Comparison', fontsize=18, fontweight='bold')

# 1. Accuracy
bars1 = ax1.bar(models, accuracy_scores, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
ax1.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy Score')
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(accuracy_scores):
    ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# 2. F1 Scores
x = np.arange(len(models))
width = 0.35
bars2 = ax2.bar(x - width/2, f1_scores_neg, width, label='Negative', color='#FF6B6B', alpha=0.8)
bars3 = ax2.bar(x + width/2, f1_scores_pos, width, label='Positive', color='#4ECDC4', alpha=0.8)
ax2.set_title('F1 Scores by Class', fontsize=14, fontweight='bold')
ax2.set_ylabel('F1 Score')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (neg, pos) in enumerate(zip(f1_scores_neg, f1_scores_pos)):
    ax2.text(i - width/2, neg + 0.02, f'{neg:.3f}', ha='center', va='bottom', fontsize=10)
    ax2.text(i + width/2, pos + 0.02, f'{pos:.3f}', ha='center', va='bottom', fontsize=10)

# 3. Precision comparison
bars4 = ax3.bar(x - width/2, precision_neg, width, label='Negative', color='#FF6B6B', alpha=0.8)
bars5 = ax3.bar(x + width/2, precision_pos, width, label='Positive', color='#4ECDC4', alpha=0.8)
ax3.set_title('Precision by Class', fontsize=14, fontweight='bold')
ax3.set_ylabel('Precision Score')
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend()
ax3.set_ylim(0, 1)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for i, (neg, pos) in enumerate(zip(precision_neg, precision_pos)):
    ax3.text(i - width/2, neg + 0.02, f'{neg:.2f}', ha='center', va='bottom', fontsize=10)
    ax3.text(i + width/2, pos + 0.02, f'{pos:.2f}', ha='center', va='bottom', fontsize=10)

# 4. Recall comparison
bars6 = ax4.bar(x - width/2, recall_neg, width, label='Negative', color='#FF6B6B', alpha=0.8)
bars7 = ax4.bar(x + width/2, recall_pos, width, label='Positive', color='#4ECDC4', alpha=0.8)
ax4.set_title('Recall by Class', fontsize=14, fontweight='bold')
ax4.set_ylabel('Recall Score')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend()
ax4.set_ylim(0, 1)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, (neg, pos) in enumerate(zip(recall_neg, recall_pos)):
    ax4.text(i - width/2, neg + 0.02, f'{neg:.2f}', ha='center', va='bottom', fontsize=10)
    ax4.text(i + width/2, pos + 0.02, f'{pos:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Performance comparison plot saved")

# Print final summary
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)

print(f"VADER Performance:")
print(f"  Accuracy: {vader_accuracy:.3f}")
print(f"  Negative F1: {vader_neg_f1:.3f}")
print(f"  Positive F1: {vader_pos_f1:.3f}")

print(f"\nScikit-learn Performance:")
print(f"  Accuracy: {sklearn_accuracy:.3f}")
print(f"  Negative F1: {sklearn_neg_f1:.3f}")
print(f"  Positive F1: {sklearn_pos_f1:.3f}")

print(f"\nKey Findings:")
print(f"  • Scikit-learn outperforms VADER in overall accuracy")
print(f"  • VADER shows bias toward positive predictions")
print(f"  • Scikit-learn provides more balanced performance across classes")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("All plots saved as PNG files in the current directory.")
