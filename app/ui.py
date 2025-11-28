import gradio as gr
import requests

# URL API FastAPI (ganti jika deploy di server lain)
API_URL = "http://localhost:8000/predict"

def predict_resign(tingkat_kepuasan, lama_bekerja, kecelakaan_kerja, gaji, jam_kerja_perbulan):
    # Siapkan payload JSON
    payload = {
        "tingkat_kepuasan": tingkat_kepuasan,
        "lama_bekerja": lama_bekerja,
        "kecelakaan_kerja": kecelakaan_kerja,
        "gaji": gaji,
        "jam_kerja_perbulan": jam_kerja_perbulan
    }
    
    try:
        # Panggil API
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise error jika gagal
        result = response.json()
        return f"Prediksi Resign: {result['prediction']}"
    except Exception as e:
        return f"Error: {str(e)}"

# Buat interface Gradio
with gr.Blocks(title="Prediksi Karyawan Resign") as demo:
    gr.Markdown("# Prediksi Karyawan Resign Menggunakan Random Forest")
    
    with gr.Row():
        tingkat_kepuasan = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Tingkat Kepuasan (0-1)")
        lama_bekerja = gr.Number(label="Lama Bekerja (tahun)", precision=0)
    
    with gr.Row():
        kecelakaan_kerja = gr.Radio(choices=[0, 1], label="Kecelakaan Kerja (0: Tidak, 1: Pernah)")
        gaji = gr.Dropdown(choices=[0, 1, 2], label="Gaji (0: Low, 1: Medium, 2: High)")
    
    jam_kerja_perbulan = gr.Number(label="Jam Kerja per Bulan", precision=0)
    
    submit_btn = gr.Button("Prediksi")
    output = gr.Textbox(label="Hasil Prediksi")
    
    submit_btn.click(
        fn=predict_resign,
        inputs=[tingkat_kepuasan, lama_bekerja, kecelakaan_kerja, gaji, jam_kerja_perbulan],
        outputs=output
    )

# Jalankan Gradio
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)