import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi Sastrawi
factory_stop = StopWordRemoverFactory()
stopword_remover = factory_stop.create_stop_word_remover()

factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Case Folding: Ubah ke huruf kecil
    text = text.lower()
    
    # 2. Data Cleaning: Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Data Cleaning: Hapus angka dan karakter selain huruf alphabet
    text = re.sub(r'[^a-z\s]', '', text)

    # ==========================================
    # 4. NORMALISASI SINONIM / EUFEMISME DENGAN REGEX
    # ==========================================
    # Jaring 1: Menangkap kombinasi kata "tanpa baju", "tidak pakai busana", "buka pakaian", dll
    # \b = batas kata, \s+ = spasi satu atau lebih
    #tamabahkan  
    pola_tanpa_baju = r'\b(tanpa|tidak pakai|gak pakai|nggak pakai|buka|lepas)\s+(baju|pakaian|busana|celana|baju dalam|pakaian dalam)\b'
    text = re.sub(pola_tanpa_baju, 'telanjang', text)
    # ==========================================
    # TAMBAHAN REVISI SIDANG DARI DOSEN PENGUJI 2
    # ==========================================
    # Jaring 2: Menyelamatkan frasa kiasan sains/aman agar tidak diblokir oleh TF-IDF
    # Jaring 2: Menyelamatkan frasa kiasan sains/aman agar tidak diblokir oleh TF-IDF
    text = re.sub(r'\bmata telanjang\b', 'penglihatan langsung', text)
    
    # Menyelamatkan konteks edukasi/sehari-hari untuk kata "keluar"
    # UBAH BAGIAN INI: Ganti "meninggalkan" menjadi "pergi dari" atau "izin dari"
    text = re.sub(r'\bkeluar kelas\b', 'pergi dari kelas', text)
    text = re.sub(r'\bkeluar rumah\b', 'pergi dari rumah', text)
    # Jaring 3: Normalisasi bahasa gaul sebelum masuk ke Sastrawi
    # Mencegah Sastrawi memotong "kepengen" menjadi "ken"
    text = re.sub(r'\bkepengen\b', 'ingin', text)
    text = re.sub(r'\bpengen\b', 'ingin', text)

    # Jaring 2: Menangkap eufemisme hubungan seksual
    pola_seks = r'\b(berhubungan badan|berhubungan intim|main serong|tidur bareng)\b'
    text = re.sub(pola_seks, 'seks', text)

    # Kata tunggal yang langsung diubah
    text = text.replace('bugil', 'telanjang')
    
    # 5. Stopword Manual: Buang kata pengantar yang mengganggu skor SVM
    kata_umum = ['buatkan', 'tolong', 'saya', 'aku', 'gambarkan', 'generate', 'untuk', 'bikinkan']
    text = ' '.join([kata for kata in text.split() if kata not in kata_umum])
    
    # 6. Filtering Sastrawi: Hapus stopwords bahasa Indonesia bawaan
    text = stopword_remover.remove(text)
    
    # 7. Stemming Sastrawi: Kembalikan kata ke bentuk dasar
    text = stemmer.stem(text)
    
    # 8. Hilangkan spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text