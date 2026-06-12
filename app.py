from flask import Flask, request, render_template
import joblib
from deep_translator import GoogleTranslator
from scripts.preprocessing import clean_text
from lime.lime_text import LimeTextExplainer
import json
import os

app = Flask(__name__)   

# Setup Path Dasar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'svm_model.pkl')
VEC_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')

try:
    svm_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    class_names = ['Aman', 'Blokir']
    lime_explainer = LimeTextExplainer(class_names=class_names)
except Exception as e:
    print(f"Error: Inisialisasi gagal. {e}")

def predict_proba_lime(texts):
    texts_vec = vectorizer.transform(texts)
    return svm_model.predict_proba(texts_vec)

@app.route('/', methods=['GET', 'POST'])
def index():
    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)

    if request.method == 'POST':
        raw_prompt = request.form.get('prompt', '')
        
        if not raw_prompt.strip():
            return render_template('index.html', error="Prompt tidak boleh kosong.", metrics=metrics)

        try:
            translator = GoogleTranslator(source='auto', target='id')
            translated_prompt = translator.translate(raw_prompt)
        except Exception:
            translated_prompt = raw_prompt 
            
        cleaned_prompt = clean_text(translated_prompt)
        
        vec_prompt = vectorizer.transform([cleaned_prompt])
        prediction = svm_model.predict(vec_prompt)[0]
        probabilities = svm_model.predict_proba(vec_prompt)[0]
        
        prob_aman = probabilities[0]
        prob_blokir = probabilities[1]
        
        if prediction == 1:
            confidence_score = round(prob_blokir * 100, 1)
            result_msg = "KONTEN DIBLOKIR: Terdeteksi unsur pornografi/narasi vulgar"
            status_class = "blocked"
        else:
            confidence_score = round(prob_aman * 100, 1)
            result_msg = "PROMPT AMAN"
            status_class = "safe"

        # === KODE BARU: AMBIL DATA MENTAH LIME LENGKAP ===
        lime_data_custom = None
        if cleaned_prompt.strip():
            try:
                explanation = lime_explainer.explain_instance(
                    cleaned_prompt, 
                    predict_proba_lime, 
                    num_features=10, 
                    labels=(1,) 
                )
                
                # Kita kemas data yang dibutuhkan untuk digambar di HTML
                lime_data_custom = {
                    "prob_aman": round(prob_aman * 100, 1),
                    "prob_blokir": round(prob_blokir * 100, 1),
                    "weights": explanation.as_list(label=1), # [(kata, bobot), ...]
                    "text": cleaned_prompt.split() # Kalimat yang dipecah jadi kata-kata
                }
            except Exception as lime_e:
                print(f"Error LIME: {lime_e}")
            
        return render_template('index.html', 
                               original=raw_prompt, 
                               translated=translated_prompt,
                               result=result_msg, 
                               status_class=status_class,
                               metrics=metrics,
                               confidence=confidence_score,
                               lime_custom=lime_data_custom) # <-- Mengirim data custom
                               
    return render_template('index.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)