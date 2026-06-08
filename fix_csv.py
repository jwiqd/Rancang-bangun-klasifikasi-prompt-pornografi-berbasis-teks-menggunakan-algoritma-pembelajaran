import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_BERMASALAH = os.path.join(BASE_DIR, 'dataset', 'dataset_tambahan_1.csv')
FILE_DIPERBAIKI = os.path.join(BASE_DIR, 'dataset', 'dataset_tambahan_fixed.csv')

print("Mulai memperbaiki file CSV dengan metode ekstra aman...")

with open(FILE_BERMASALAH, 'r', encoding='utf-8') as f_in, open(FILE_DIPERBAIKI, 'w', encoding='utf-8', newline='') as f_out:
    # Menggunakan modul csv bawaan Python agar penulisan formatnya standar
    writer = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
    
    for i, line in enumerate(f_in):
        line = line.strip()
        if not line: continue
        
        # Penanganan Header (Baris Pertama)
        if i == 0 and 'text' in line.lower() and 'label' in line.lower():
            writer.writerow(['text', 'label'])
            continue
            
        # Cari letak koma paling terakhir (pemisah antara teks dan positif/negatif)
        last_comma_idx = line.rfind(',')
        
        if last_comma_idx != -1:
            text_part = line[:last_comma_idx].strip()
            label_part = line[last_comma_idx+1:].strip()
            
            # KUNCI PERBAIKAN: Hapus semua tanda kutip dua (") di dalam teks 
            # agar tidak merusak kolom CSV
            text_part = text_part.replace('"', '')
            text_part = text_part.replace("'", "") # Hapus kutip satu juga untuk aman
            
            # Tulis ke file baru dengan format yang sempurna
            writer.writerow([text_part, label_part])
        else:
            # Jika baris berantakan tanpa koma, tulis ulang apa adanya
            writer.writerow([line])

print("Selesai! File dataset_tambahan_fixed.csv sudah bersih 100%.")