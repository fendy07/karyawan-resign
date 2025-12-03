from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import gradio as gr
import numpy as np
import pandas as pd
import os
import sys

# Handle imports for both local and Docker environments
try:
    from .utils import logger, log_prediction, log_error, log_info, log_debug
except ImportError:
    from utils import logger, log_prediction, log_error, log_info, log_debug

# Get the base directory for model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.joblib')

# Load model dan scaler (ganti path jika perlu)
try:
    log_info("Loading machine learning models...")
    log_info(f"Model path: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    log_info("Random Forest model loaded successfully")
    
    log_info(f"Scaler path: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    log_info("Scaler loaded successfully")
except Exception as e:
    log_error(f"Failed to load models: {str(e)}", exc_info=True)
    raise

app = FastAPI(title="Prediksi Karyawan Resign API")
log_info("FastAPI application initialized: Prediksi Karyawan Resign API")

# Definisikan schema input
class InputData(BaseModel):
    tingkat_kepuasan: float
    lama_bekerja: int
    kecelakaan_kerja: int
    gaji: int
    jam_kerja_perbulan: int

# Endpoint untuk prediksi
@app.post("/predict")
async def predict_resign(data: InputData):
    try:
        log_info(f"Received prediction request with data: {data.dict()}")
        
        # Konversi input ke DataFrame
        input_df = pd.DataFrame([data.dict()])
        log_debug("Input data converted to DataFrame")

        # Scaling input
        input_scaled = scaler.transform(input_df)
        log_debug("Input data scaled successfully")

        # Prediksi dengan probability
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = max(probabilities) * 100  # Konversi ke persentase
        resign_status = "ya" if prediction == 1 else "tidak"
        
        log_prediction(data.dict(), resign_status, confidence, probabilities)

        return {
            "prediction": resign_status,
            "confidence": round(confidence, 2),
            "probability_tidak_resign": round(probabilities[0] * 100, 2),
            "probability_resign": round(probabilities[1] * 100, 2)
        }
    except Exception as e:
        log_error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Tambahkan bagian ini di akhir untuk integrasi Gradio ---
# Fungsi prediksi langsung untuk Gradio (menggunakan model secara langsung)
def predict_resign_direct(tingkat_kepuasan, lama_bekerja, kecelakaan_kerja, gaji, jam_kerja_perbulan):
    try:
        input_data = {
            "tingkat_kepuasan": tingkat_kepuasan,
            "lama_bekerja": lama_bekerja,
            "kecelakaan_kerja": kecelakaan_kerja,
            "gaji": gaji,
            "jam_kerja_perbulan": jam_kerja_perbulan
        }
        log_info(f"Gradio prediction request: {input_data}")
        
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = max(probabilities) * 100
        resign_status = 'ya' if prediction == 1 else 'tidak'
        
        log_prediction(input_data, resign_status, confidence, probabilities)
        
        # Hitung progress bar untuk probabilitas
        prob_tidak_resign = probabilities[0] * 100
        prob_resign = probabilities[1] * 100
        
        # Buat visualisasi detail probabilitas
        detail_output = f"""
📊 HASIL PREDIKSI RESIGN KARYAWAN

🎯 PREDIKSI AKHIR: {resign_status.upper()}
💯 CONFIDENCE SCORE: {confidence:.2f}%

📈 DETAIL PROBABILITAS

❌ Tidak Resign: {prob_tidak_resign:.2f}%
   {'█' * int(prob_tidak_resign / 5)}{'░' * (20 - int(prob_tidak_resign / 5))}

✅ Resign: {prob_resign:.2f}%
   {'█' * int(prob_resign / 5)}{'░' * (20 - int(prob_resign / 5))}

📋 DATA INPUT

• Tingkat Kepuasan: {tingkat_kepuasan:.2f}
• Lama Bekerja: {int(lama_bekerja)} tahun
• Kecelakaan Kerja: {'Pernah' if kecelakaan_kerja == 1 else 'Tidak Pernah'}
• Level Gaji: {['Low', 'Medium', 'High'][int(gaji)]}
• Jam Kerja per Bulan: {int(jam_kerja_perbulan)} jam

🔍 Analisis:
• Karyawan dengan tingkat kepuasan rendah cenderung memiliki risiko resign lebih tinggi.
• Lama bekerja yang singkat juga dapat meningkatkan kemungkinan resign.
• Kecelakaan kerja dapat mempengaruhi keputusan karyawan untuk resign.
• Gaji yang lebih tinggi biasanya berbanding terbalik dengan tingkat resign.
• Jam kerja yang terlalu banyak dapat menyebabkan kelelahan dan meningkatkan risiko resign.

"""
        return detail_output

    except Exception as e:
        log_error(f"Error in Gradio prediction: {str(e)}", exc_info=True)
        return f"❌ Error: {str(e)}"

# Buat interface Gradio
gradio_interface = gr.Interface(
    fn=predict_resign_direct,
    inputs=[
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Tingkat Kepuasan (0-1)"),
        gr.Number(label="Lama Bekerja (tahun)", precision=0),
        gr.Radio(choices=[0, 1], label="Kecelakaan Kerja (0: Tidak, 1: Pernah)"),
        gr.Dropdown(choices=[0, 1, 2], label="Gaji (0: Low, 1: Medium, 2: High)"),
        gr.Number(label="Jam Kerja per Bulan", precision=0)
    ],
    outputs=gr.Textbox(label="Hasil Prediksi", lines=12),
    title="Prediksi Karyawan Resign Menggunakan Algoritma Random Forest",
    description="Masukkan data karyawan untuk memprediksi kemungkinan resign."
)

gr.Markdown("""
            ### 💡 Tips:
            - Gunakan slider dan input yang sesuai untuk mendapatkan prediksi terbaik.
            - Pastikan data yang dimasukkan akurat dan relevan.
            - Gradio interface siap diakses di /gradio
            - Confidence score menunjukkan seberapa yakin model terhadap prediksi yang diberikan.
            """)

gr.Markdown("""
            ----
            **Fendy Hendriyanto - AI Mentor | Version 1.0.0 | © 2025 
            """)

# Mount Gradio ke FastAPI di route /gradio
log_info("Mounting Gradio interface at /gradio")
gr.mount_gradio_app(app, gradio_interface, path="/gradio")
log_info("Gradio interface mounted successfully")

# Jalankan dengan: uvicorn app:app --reload
if __name__ == "__main__":
    log_info("Starting Uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)