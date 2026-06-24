import pandas as pd
import os

FILE_1 = 'dataset/dataset500.csv' 
FILE_2 = 'dataset/dataset_tambahan_fixed_cleaned.csv'
FILE_OUTPUT = 'dataset/dataset_final.csv'

def standarisasi_label(val):
    """
    Fungsi cerdas untuk menerjemahkan label yang bercampur
    menjadi format baku: 0 (Positif/Aman) dan 1 (Negatif/Blokir).
    """
    val_str = str(val).strip().lower()
    
    # label aman diberi nilai 0
    if val_str in ['0', '0.0', 'aman', 'positif', 'positive']:
        return 0
    # label blokir diberi nilai 1
    elif val_str in ['1', '1.0', 'blokir', 'negatif', 'negative']:
        return 1
    else:
        
        return None

print("Memulai proses penggabungan dataset...")

try:
    # 1. Buka kedua dataset
    # Mengambil hanya 2 kolom pertama untuk menghindari kolom ekstra yang tidak sengaja terbuat
    print(f"Membaca {FILE_1}...")
    df1 = pd.read_csv(FILE_1, header=None, usecols=[0, 1], names=['teks', 'label'], skiprows=1)
    
    print(f"Membaca {FILE_2}...")
    df2 = pd.read_csv(FILE_2, header=None, usecols=[0, 1], names=['teks', 'label'], skiprows=1)

    # 2. Gabungkan kedua dataset secara vertikal
    df_gabungan = pd.concat([df1, df2], ignore_index=True)
    print(f"Total baris awal setelah digabung: {len(df_gabungan)} baris.")

    # 3. Terapkan fungsi standarisasi label
    df_gabungan['label'] = df_gabungan['label'].apply(standarisasi_label)

    # 4. Pembersihan Data (Data Cleaning)
    # Hapus baris yang labelnya tidak valid (None) atau teksnya kosong
    df_gabungan = df_gabungan.dropna(subset=['teks', 'label'])
    
    # Hapus duplikat (jika ada kalimat yang sama persis di kedua dataset)
    df_gabungan = df_gabungan.drop_duplicates(subset=['teks'])
    
    # Pastikan label berformat integer murni (0 atau 1)
    df_gabungan['label'] = df_gabungan['label'].astype(int)

    # 5. Simpan ke file baru
    df_gabungan.to_csv(FILE_OUTPUT, index=False)
    
    print("\n✅ BERHASIL!")
    print(f"Dataset final tersimpan di: {FILE_OUTPUT}")
    print(f"Total baris bersih yang siap dilatih: {len(df_gabungan)} baris.")
    print("Distribusi Label:")
    print(df_gabungan['label'].value_counts())

except Exception as e:
    print(f"\n❌ Terjadi kesalahan: {e}")