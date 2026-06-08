from flask import Flask, request, render_template
import joblib
from deep_translator import GoogleTranslator
from scripts.preprocessing import clean_text
import os

app = Flask(__name__)

# Setup Path dan Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'svm_model.pkl')
VEC_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')

try:
    svm_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
except Exception as e:
    print(f"Error: Model belum di-training atau path salah. {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        raw_prompt = request.form.get('prompt', '')
        
        if not raw_prompt.strip():
            return render_template('index.html', error="Prompt tidak boleh kosong.")

        # Modul Translasi: 'auto' mendeteksi sumber bahasa otomatis, targetkan ke 'id' (Indonesia)
        try:
            translator = GoogleTranslator(source='auto', target='id')
            translated_prompt = translator.translate(raw_prompt)
        except Exception:
            translated_prompt = raw_prompt # Fallback jika koneksi translator gagal
            
        # Pipeline Preprocessing
        cleaned_prompt = clean_text(translated_prompt)
        
        # Prediksi SVM
        vec_prompt = vectorizer.transform([cleaned_prompt])
        prediction = svm_model.predict(vec_prompt)[0]
        
        # Logika Output
        if prediction == 1:
            result_msg = "KONTEN DIBLOKIR: Terdeteksi unsur pornografi/narasi vulgar"
            status_class = "blocked"
        else:
            result_msg = "PROMPT AMAN"
            status_class = "safe"
            
        return render_template('index.html', 
                               original=raw_prompt, 
                               translated=translated_prompt,
                               result=result_msg, 
                               status_class=status_class)
                               
    # Request method GET
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)