import nltk
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Download VADER lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Test VADER
a = "This is a good movie"
print("Example 1:", a)
print("Sentiment:", sid.polarity_scores(a))

b = "This was the best, awesome movie EVER MADE!!!"
print("\nExample 2:", b)
print("Sentiment:", sid.polarity_scores(b))

c = "This was the worst movie that has disgraced the screen."
print("\nExample 3:", c)
print("Sentiment:", sid.polarity_scores(c))

# Load data
df = pd.read_csv('amazonreviews.tsv', sep='\t')
print(f"\nDataset shape: {df.shape}")

# Data cleaning
df.dropna(inplace=True)
blanks = []
for i, lb, rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)
df.drop(blanks, inplace=True)

# VADER analysis
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda d: d['compound'])
df['comp_score'] = df['compound'].apply(lambda score: 'pos' if score >= 0 else 'neg')

vader_accuracy = accuracy_score(df['label'], df['comp_score'])
print(f"\nVADER Accuracy: {vader_accuracy:.4f}")
print("\nVADER Classification Report:")
print(classification_report(df['label'], df['comp_score']))

# Scikit-learn analysis
X = df["review"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC(random_state=42))
])

text_clf.fit(X_train, y_train)
predictions_pipeline = text_clf.predict(X_test)

sklearn_accuracy = accuracy_score(y_test, predictions_pipeline)
print(f"\nScikit-learn Accuracy: {sklearn_accuracy:.4f}")
print("\nScikit-learn Classification Report:")
print(classification_report(y_test, predictions_pipeline))

# Extract metrics for comparison
vader_report = classification_report(df['label'], df['comp_score'], output_dict=True)
sklearn_report = classification_report(y_test, predictions_pipeline, output_dict=True)

vader_neg_precision = vader_report['neg']['precision']
vader_pos_precision = vader_report['pos']['precision']
vader_neg_recall = vader_report['neg']['recall']
vader_pos_recall = vader_report['pos']['recall']

sklearn_neg_precision = sklearn_report['neg']['precision']
sklearn_pos_precision = sklearn_report['pos']['precision']
sklearn_neg_recall = sklearn_report['neg']['recall']
sklearn_pos_recall = sklearn_report['pos']['recall']

# Create visualization
models = ['VADER', 'Scikit-learn']
accuracy_scores = [vader_accuracy, sklearn_accuracy]
precision_neg = [vader_neg_precision, sklearn_neg_precision]
precision_pos = [vader_pos_precision, sklearn_pos_precision]
recall_neg = [vader_neg_recall, sklearn_neg_recall]
recall_pos = [vader_pos_recall, sklearn_pos_recall]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('VADER vs Scikit-learn Sentiment Analysis Comparison', fontsize=16, fontweight='bold')

# Overall Accuracy
bars1 = ax1.bar(models, accuracy_scores, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy Score')
ax1.set_ylim(0, 1)
for i, v in enumerate(accuracy_scores):
    ax1.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Precision Comparison
x = np.arange(len(models))
width = 0.35
bars2 = ax2.bar(x - width/2, precision_neg, width, label='Negative', color='#FF6B6B', alpha=0.8)
bars3 = ax2.bar(x + width/2, precision_pos, width, label='Positive', color='#4ECDC4', alpha=0.8)
ax2.set_title('Precision by Class', fontsize=14, fontweight='bold')
ax2.set_ylabel('Precision Score')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.set_ylim(0, 1)

# Recall Comparison
bars4 = ax3.bar(x - width/2, recall_neg, width, label='Negative', color='#FF6B6B', alpha=0.8)
bars5 = ax3.bar(x + width/2, recall_pos, width, label='Positive', color='#4ECDC4', alpha=0.8)
ax3.set_title('Recall by Class', fontsize=14, fontweight='bold')
ax3.set_ylabel('Recall Score')
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend()
ax3.set_ylim(0, 1)

# Summary Table
ax4.axis('tight')
ax4.axis('off')
table_data = [
    ['Model', 'Accuracy', 'Neg Precision', 'Pos Precision', 'Neg Recall', 'Pos Recall'],
    ['VADER', f'{vader_accuracy:.2f}', f'{vader_neg_precision:.2f}', f'{vader_pos_precision:.2f}', f'{vader_neg_recall:.2f}', f'{vader_pos_recall:.2f}'],
    ['Scikit-learn', f'{sklearn_accuracy:.2f}', f'{sklearn_neg_precision:.2f}', f'{sklearn_pos_precision:.2f}', f'{sklearn_neg_recall:.2f}', f'{sklearn_pos_recall:.2f}']
]
table = ax4.table(cellText=table_data[1:], colLabels=table_data[0], 
                  cellLoc='center', loc='center', colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

# Add value labels
for ax in [ax2, ax3]:
    if ax == ax2:
        neg_data, pos_data = precision_neg, precision_pos
    else:
        neg_data, pos_data = recall_neg, recall_pos
    
    for i in range(len(models)):
        ax.text(i - width/2, neg_data[i] + 0.01, f'{neg_data[i]:.2f}', ha='center', va='bottom', fontsize=10)
        ax.text(i + width/2, pos_data[i] + 0.01, f'{pos_data[i]:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
