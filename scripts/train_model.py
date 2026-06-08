import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Definisikan path absolut berdasarkan lokasi file ini berjalan
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 1. LOAD DATASET (MEMBACA 2 FILE CSV)
# ==========================================
DATASET_LAMA_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset500.csv')
DATASET_BARU_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset_tambahan_fixed_cleaned.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print("Loading data...")
# Baca kedua file
df_lama = pd.read_csv(DATASET_LAMA_PATH)
df_baru = pd.read_csv(DATASET_BARU_PATH)

# Gabungkan kedua file menjadi satu tabel besar di memori
df = pd.concat([df_lama, df_baru], ignore_index=True)

# Tampilkan informasi ke terminal untuk memastikan jumlahnya benar
print(f"Total data setelah digabung: {len(df)} baris")

# Bersihkan nilai kosong (NaN) jika ada
df = df.dropna(subset=['text', 'label'])

# LOGIKA LABEL ENCODING:
# 'negatif' (pornografi/vulgar) dipetakan menjadi 1 (Kelas Positif/Target yang dicari)
# 'positif' (aman) dipetakan menjadi 0 (Kelas Negatif)
df['label_encoded'] = df['label'].apply(lambda x: 1 if str(x).strip().lower() == 'negatif' else 0)

# Preprocessing Dataset
from preprocessing import clean_text
print("Preprocessing teks... (ini mungkin memakan waktu bergantung ukuran data)")
df['teks_bersih'] = df['text'].apply(clean_text)

# Ekstraksi Fitur TF-IDF
# Menggunakan ngram_range=(1,2) untuk menangkap konteks gabungan 2 kata (intent/frasa), bukan sekadar keyword tunggal.
print("Mengekstraksi fitur TF-IDF...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['teks_bersih'])
y = df['label_encoded']

# Splitting Data (80:20) dengan Shuffling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Training SVM Linear Kernel
print("Melatih model SVM...")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)

# Evaluasi
y_pred = svm_model.predict(X_test)
print("\n=== HASIL EVALUASI ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAkurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# PENJELASAN KRUSIAL F1-SCORE:
# Dalam sistem keamanan, menyeimbangkan Precision dan Recall sangat penting. 
# Jika dataset Anda *imbalanced* (misal: lebih banyak prompt aman daripada pornografi), 
# metrik Akurasi bisa menyesatkan. F1-Score memberikan metrik yang lebih solid untuk 
# menilai seberapa baik model mendeteksi kelas target (1) tanpa menghasilkan terlalu 
# banyak false positive (memblokir instruksi yang sebenarnya aman).

# Simpan Model dan Vectorizer
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

joblib.dump(svm_model, os.path.join(MODELS_DIR, 'svm_model.pkl'))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
print(f"\nModel dan Vectorizer berhasil disimpan di folder: {MODELS_DIR}")