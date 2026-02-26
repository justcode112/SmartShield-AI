# ==========================================
# SmartShield Pro - Hybrid AI Spam Detection
# Author: Numan
# ==========================================

import pandas as pd
import re
import string
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ----------------------------
# 2. Clean Text
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['message'] = df['message'].apply(clean_text)

# ----------------------------
# 3. Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# ----------------------------
# 4. Vectorizer
# ----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words='english',
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# 5. Logistic Regression
# ----------------------------
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)

model.fit(X_train_vec, y_train)

# ----------------------------
# 6. Evaluation Metrics
# ----------------------------
y_pred = model.predict(X_test_vec)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - SmartShield")
plt.savefig("confusion_matrix.png")
plt.close()

# ----------------------------
# 7. Feature Importance
# ----------------------------
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_spam = np.argsort(coefficients)[-20:]
top_ham = np.argsort(coefficients)[:20]

plt.figure(figsize=(10,6))
plt.barh(feature_names[top_spam], coefficients[top_spam])
plt.title("Top 20 Spam Indicators")
plt.savefig("top_spam_words.png")
plt.close()

plt.figure(figsize=(10,6))
plt.barh(feature_names[top_ham], coefficients[top_ham])
plt.title("Top 20 Ham Indicators")
plt.savefig("top_ham_words.png")
plt.close()

# ----------------------------
# 8. Save Model
# ----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel & visualizations saved successfully!")