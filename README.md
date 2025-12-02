# <b>Prediksi Karyawan Resign</b>

Proyek ini adalah aplikasi machine learning untuk memprediksi kemungkinan karyawan resign berdasarkan data HR. Model yang digunakan adalah Decision Tree dan Random Forest, dilatih menggunakan dataset `Dataset_HR.csv`. Aplikasi di-deploy menggunakan FastAPI sebagai backend API dan Gradio sebagai frontend UI. Selain itu, mendukung deployment dengan Docker untuk kemudahan skalabilitas.

## Deskripsi

- **Fitur Utama**:
  - Prediksi resign karyawan berdasarkan input: `tingkat kepuasan`, `lama bekerja`, `kecelakaan kerja`, `gaji`, dan `jam kerja per bulan`.
  - Model: Random Forest (default) atau Decision Tree.
  - API Endpoint: `/predict` untuk prediksi via JSON.
  - UI: Gradio di `/gradio` untuk input interaktif.
- **Dataset**: Berdasarkan `Dataset_HR.csv` (tidak disertakan di repo ini; asumsikan Anda punya).
- **Bahasa**: Python 3.10+.

## Persyaratan

- Python 3.10 atau lebih tinggi.
- Dependensi (lihat `requirements.txt`):

  ```text
  fastapi
  uvicorn
  gradio
  joblib
  pandas
  numpy
  scikit-learn
  ```

- Docker (untuk deployment containerized).

## Instalasi

1. Clone repository (jika ada repo; atau buat folder project).
2. Instal dependensi:

   ```bash
   pip install -r requirements.txt
   ```

3. Pastikan folder `models/` berisi:
   - `random_forest_model.joblib`
   - `scaler.joblib`
   (Dihasilkan dari `train_model.py`).

## Cara Menjalankan Lokal

1. Jalankan training model (opsional, jika belum ada model):

   ```bash
   python train_model.py
   ```

2. Jalankan aplikasi:

   ```bash
   uvicorn app:app --reload
   ```

   - Akses API: `http://localhost:8000/docs` (Swagger UI untuk tes endpoint `/predict`).
   - Akses UI: `http://localhost:8000/gradio` (Gradio interface).

### Contoh Penggunaan API

Kirim POST request ke `/predict` dengan JSON body:

```json
{
  "tingkat_kepuasan": 0.5,
  "lama_bekerja": 3,
  "kecelakaan_kerja": 0,
  "gaji": 1,
  "jam_kerja_perbulan": 200
}
```

Respons: `{"prediction": "ya"}` atau `"tidak"`.

### Contoh Penggunaan UI Gradio

- Buka `http://localhost:8000/gradio`.
- Masukkan nilai via slider/number/radio/dropdown.
- Klik "Submit" untuk lihat prediksi.

## Deployment dengan Docker

1. Build image:

   ```bash
   docker build -t prediksi-resign-app .
   ```

2. Run container:

   ```bash
   docker run -p 8000:8000 prediksi-resign-app
   ```

3. Akses seperti lokal: `http://localhost:8000/gradio`.

Gunakan Docker Compose jika perlu (lihat `docker-compose.yml` jika dibuat).

## Struktur Folder

```
project/
├── app.py                  # FastAPI + Gradio app
├── train_model.py          # Script training model
├── models/                 # Folder model dan scaler
│   ├── random_forest_model.joblib
│   └── scaler.joblib
├── requirements.txt        # Dependensi
├── Dockerfile              # Untuk Docker build
└── README.md               # Dokumentasi ini
```

## Catatan

- **Model Alternatif**: Ganti ke Decision Tree di `app.py` jika diinginkan (gunakan `pickle` untuk load `.pkl`).
- **Error Handling**: Pastikan input valid; app punya try-except dasar.
- **Improvement**: Tambah autentikasi untuk production, atau integrasi database untuk data real-time.
- **Kontribusi**: Fork repo dan submit PR jika ada perbaikan.

Jika ada pertanyaan, hubungi pengguna repositori ini via email ke <hendriyantofendy07@gmail.com>. Terima kasih!
