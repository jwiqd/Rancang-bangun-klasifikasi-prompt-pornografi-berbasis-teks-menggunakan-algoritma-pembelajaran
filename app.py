from flask import Flask, request, render_template, session
import joblib
from deep_translator import GoogleTranslator
from scripts.preprocessing import clean_text
from lime.lime_text import LimeTextExplainer
import json
import os

app = Flask(__name__)   
app.secret_key = 'kunci_rahasia_sidang_ta'

# Setup Path Dasar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load Kedua Model Sekaligus saat server menyala
try:
    # 1. Model SVM
    svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))
    vectorizer_svm = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    
    # 2. Model Decision Tree (DT)
    # dt_model = joblib.load(os.path.join(MODELS_DIR, 'dt_model.pkl'))
    # vectorizer_dt = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer_dt.pkl'))
    
    class_names = ['Aman', 'Blokir']
    lime_explainer = LimeTextExplainer(class_names=class_names)
except Exception as e:
    print(f"Error: Inisialisasi model gagal. Pastikan file pkl ada! {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        raw_prompt = request.form.get('prompt', '')
        # Tangkap pilihan model dari HTML (default ke SVM jika kosong)
        model_choice = request.form.get('model_type', 'svm')
        
        if not raw_prompt.strip():
            return render_template('index.html', chat_history=session.get('chat_history', []))

        #untuk memilih model yang aktif berdasarkan input pengguna
        # if model_choice == 'dt':
        #     active_model = dt_model
        #     active_vectorizer = vectorizer_dt
        #     model_name_display = "Decision Tree"
        # else:
        #     active_model = svm_model
        #     active_vectorizer = vectorizer_svm
        #     model_name_display = "SVM"
        active_model = svm_model
        active_vectorizer = vectorizer_svm
        model_name_display = "SVM"

        try:
            translator = GoogleTranslator(source='auto', target='id')
            translated_prompt = translator.translate(raw_prompt)
        except Exception:
            translated_prompt = raw_prompt 
            
        cleaned_prompt = clean_text(translated_prompt)
        
        # Gunakan Vectorizer dan Model yang sedang aktif
        vec_prompt = active_vectorizer.transform([cleaned_prompt])
        probabilities = active_model.predict_proba(vec_prompt)[0]
        
        prob_aman = probabilities[0]
        prob_blokir = probabilities[1]
        
        # Logika Threshold nilai minimal 70%
        if prob_aman >= 0.70:
            result_msg = "PROMPT AMAN"
            status_class = "safe"
            threshold_msg = f"✅ Prompt aman tidak mengandung unsur pornografi "
        else:
            result_msg = "KONTEN DIBLOKIR"
            status_class = "blocked"
            threshold_msg = f"❌ Prompt Diblokir"

        # Ekstraksi LIME (Harus menggunakan model yang aktif)
        def predict_proba_active(texts):
            texts_vec = active_vectorizer.transform(texts)
            return active_model.predict_proba(texts_vec)

        lime_data_custom = None
        if cleaned_prompt.strip():
            try:
                # Ekstraksi LIME untuk mendapatkan bobot per kata
                explanation = lime_explainer.explain_instance(
                    cleaned_prompt, 
                    predict_proba_active, 
                    num_features=10, 
                    labels=(1,) 
                )
                # Menyimpan hasil probabilitas dan bobot kata untuk dikirim ke UI
                lime_data_custom = {
                    "prob_aman": round(prob_aman * 100, 1),
                    "prob_blokir": round(prob_blokir * 100, 1),
                    "weights": explanation.as_list(label=1),
                    "text": cleaned_prompt.split() 
                }
            except Exception as lime_e:
                print(f"Error LIME: {lime_e}")
            
        # Kemas hasil, tambahkan 'model_name' agar tampil di UI
        current_chat = {
            'original': raw_prompt,
            'translated': translated_prompt,
            'result': result_msg,
            'status_class': status_class,
            'lime_custom': lime_data_custom,
            'threshold_msg': threshold_msg,
            'model_name': model_name_display 
        }

        history = session['chat_history']
        history.append(current_chat)

        if len(history) > 5:
            history = history[-5:]

        session['chat_history'] = history
        session.modified = True

        return render_template('index.html', chat_history=history) 
                               
    return render_template('index.html', chat_history=session.get('chat_history', []))

if __name__ == '__main__':
    app.run(debug=True)