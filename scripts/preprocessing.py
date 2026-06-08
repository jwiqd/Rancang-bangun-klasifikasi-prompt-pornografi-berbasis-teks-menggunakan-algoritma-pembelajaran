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
    # 4. NORMALISASI SINONIM / EUFEMISME (WAJIB DI SINI)
    # Dilakukan sebelum Sastrawi membuang kata "tanpa"
    # ==========================================
    text = text.replace('tanpa busana', 'telanjang')
    text = text.replace('tidak pakai baju', 'telanjang')
    text = text.replace('tidak berbusana', 'telanjang')
    text = text.replace('berhubungan badan', 'seks')
    
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