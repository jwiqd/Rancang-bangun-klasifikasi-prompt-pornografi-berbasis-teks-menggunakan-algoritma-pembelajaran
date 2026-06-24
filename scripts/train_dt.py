import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset_final.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 1. Load Dataset
print("Memuat dataset final...")
df = pd.read_csv(DATASET_PATH)

X = df['teks'].astype(str)
y = df['label']

# Pastikan semua label menjadi angka bulat
y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

# 2. Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Ekstraksi Fitur (TF-IDF)
print("Melakukan ekstraksi fitur TF-IDF...")
vectorizer_dt = TfidfVectorizer(ngram_range=(1, 3))
X_train_vec = vectorizer_dt.fit_transform(X_train)
X_test_vec = vectorizer_dt.transform(X_test)

# 4. Pelatihan Model DECISION TREE
print("Melatih model Decision Tree...")
# Kita atur kedalaman maksimal (max_depth) agar tidak overfitting
dt_model = DecisionTreeClassifier(class_weight='balanced', max_depth=50, random_state=42)
dt_model.fit(X_train_vec, y_train)

# 5. Evaluasi Model
y_pred = dt_model.predict(X_test_vec)
print("\n=== HASIL EVALUASI DECISION TREE ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAkurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Simpan Model Decision Tree (Optional, jika nanti ingin digabungkan ke web)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

joblib.dump(dt_model, os.path.join(MODELS_DIR, 'dt_model.pkl'))
joblib.dump(vectorizer_dt, os.path.join(MODELS_DIR, 'tfidf_vectorizer_dt.pkl'))
print(f"\nModel Decision Tree berhasil disimpan!")