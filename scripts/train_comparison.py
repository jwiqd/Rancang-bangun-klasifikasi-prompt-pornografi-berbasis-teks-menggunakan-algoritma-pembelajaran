import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset_final.csv')

# 1. Load Dataset
print("Memuat dataset final...")
df = pd.read_csv(DATASET_PATH)

X = df['teks'].astype(str)
y = df['label']

# Pastikan format label sama seperti sebelumnya
y = y.replace({'negatif': 1, 'positif': 0, '1': 1, '0': 0})
y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

# 2. Split Data (70% Training, 30% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Ekstraksi Fitur (TF-IDF)
print("Melakukan ekstraksi fitur TF-IDF...")
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================
# PELATIHAN MODEL NAIVE BAYES
# ==========================================
print("\n" + "="*40)
print("HASIL EVALUASI NAIVE BAYES")
print("="*40)
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)

print("Confusion Matrix Naive Bayes:\n", confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report Naive Bayes:\n", classification_report(y_test, y_pred_nb, digits=4))

# ==========================================
# PELATIHAN MODEL LOGISTIC REGRESSION
# ==========================================
print("\n" + "="*40)
print("HASIL EVALUASI LOGISTIC REGRESSION")
print("="*40)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)

print("Confusion Matrix Logistic Regression:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report Logistic Regression:\n", classification_report(y_test, y_pred_lr, digits=4))