import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json

# IMPORT TAMBAHAN UNTUK VISUALISASI GRAFIK
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Pastikan menggunakan dataset yang sudah kita bersihkan sebelumnya
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset_final.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 1. Load Dataset
print("Memuat dataset...")
df = pd.read_csv(DATASET_PATH)

# Pisahkan fitur teks dan label
X = df['teks'].astype(str)
y = df['label']

# Ubah label teks menjadi angka: 1 untuk negatif (Blokir), 0 untuk positif (Aman)
y = y.replace({'negatif': 1, 'positif': 0, '1': 1, '0': 0})

# Paksa semua label menjadi angka bulat (integer) agar aman dibaca oleh SVM
y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
# ----------------------------

# 2. Split Data (70% Training, 30% Testing)
# test size = 0.3 dirubah sesuai arahan dosen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Ekstraksi Fitur (TF-IDF)
print("Melakukan ekstraksi fitur TF-IDF...")
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Pelatihan Model SVM
print("Melatih model SVM...")
# class weight balance dihapus sesuai arahan dosen karena data sudah seimbang
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_vec, y_train)

# 5. Evaluasi Model
y_pred = svm_model.predict(X_test_vec)
print("\n=== HASIL EVALUASI ===")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nAkurasi:", accuracy_score(y_test, y_pred))

# Ambil data lengkap dalam format dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. SIMPAN METRIK KE JSON UNTUK WEBSITE
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

metrics_data = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": report_dict["weighted avg"]["precision"],
    "recall": report_dict["weighted avg"]["recall"],
    "f1_score": report_dict["weighted avg"]["f1-score"],
    "support": report_dict["weighted avg"]["support"]
}

# Simpan ke metrics.json
with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics_data, f)

# 7. Simpan Model dan Vectorizer
joblib.dump(svm_model, os.path.join(MODELS_DIR, 'svm_model.pkl'))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
print(f"\nModel, Vectorizer, dan Metrik berhasil disimpan di folder: {MODELS_DIR}")

# ==========================================
# 8. MENAMPILKAN VISUALISASI CONFUSION MATRIX
# ==========================================
print("\nMenampilkan grafik Confusion Matrix... (Tutup jendela grafik untuk mengakhiri program)")
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Support Vector Machine (SVM)')
plt.ylabel('Label Asli (Aktual)')
plt.xlabel('Tebakan Model (Prediksi)')
plt.show()