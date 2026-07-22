import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Masukkan data matriks dari hasil pengujian terminal Anda
cm_nb = np.array([[167, 5], 
                  [3, 187]])

cm_lr = np.array([[168, 4], 
                  [9, 181]])

# Nama label untuk sumbu X dan Y
class_names = ['Aman (0)', 'Vulgar (1)']

# ==========================================
# VISUALISASI NAIVE BAYES
# ==========================================
plt.figure(figsize=(6, 4))
# Saya gunakan warna 'Greens' (Hijau) agar berbeda dari SVM (Biru)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 14}) # Memperbesar angka di dalam kotak

plt.title('Confusion Matrix - Naive Bayes', fontsize=14, pad=15)
plt.ylabel('Label Aktual (Asli)', fontsize=12)
plt.xlabel('Label Prediksi', fontsize=12)
plt.tight_layout()

# Menyimpan gambar
plt.savefig('cm_naive_bayes.png', dpi=300)
print("Gambar cm_naive_bayes.png berhasil disimpan!")
plt.show()

# ==========================================
# VISUALISASI LOGISTIC REGRESSION
# ==========================================
plt.figure(figsize=(6, 4))
# Saya gunakan warna 'Oranges' (Oranye) agar bervariasi
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 14}) 

plt.title('Confusion Matrix - Logistic Regression', fontsize=14, pad=15)
plt.ylabel('Label Aktual (Asli)', fontsize=12)
plt.xlabel('Label Prediksi', fontsize=12)
plt.tight_layout()

# Menyimpan gambar
plt.savefig('cm_logistic_regression.png', dpi=300)
print("Gambar cm_logistic_regression.png berhasil disimpan!")
plt.show()