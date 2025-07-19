# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model yang sudah disimpan (TIDAK PERLU scaler)
model = joblib.load('model_depresi.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cek')
def cek():
    return render_template('cek.html')

# Ganti fungsi predict Anda dengan yang ini
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- Proses Prediksi (Selalu Dijalankan) ---

        # Ambil semua jawaban dari form
        jawaban_user_dict = request.form

        # Siapkan data untuk model
        jawaban_user_list = [
            float(jawaban_user_dict.get('gender')),
            float(jawaban_user_dict.get('age')),
            float(jawaban_user_dict.get('academic_pressure')),
            float(jawaban_user_dict.get('work_pressure')),
            (float(jawaban_user_dict.get('ipk')) / 4) * 10,  # Konversi IPK
            float(jawaban_user_dict.get('study_satisfaction')),
            float(jawaban_user_dict.get('job_satisfaction')),
            float(jawaban_user_dict.get('sleep_duration')),
            float(jawaban_user_dict.get('dietary_habits')),
            float(jawaban_user_dict.get('degree')),
            float(jawaban_user_dict.get('suicidal_thoughts')),
            float(jawaban_user_dict.get('work_study_hours')),
            float(jawaban_user_dict.get('financial_stress')),
            float(jawaban_user_dict.get('family_history'))
        ]

        # Lakukan prediksi seperti biasa
        fitur = [np.array(jawaban_user_list)]
        probabilitas = model.predict_proba(fitur)
        prob_depresi = probabilitas[0][1]
        persentase_depresi = round(prob_depresi * 100, 2)

        # Siapkan teks hasil prediksi
        if persentase_depresi > 50:
            hasil_teks = f"Berdasarkan jawaban Anda, sistem memprediksi ada {persentase_depresi}% kemungkinan Anda mengalami gejala yang mengarah pada depresi."
        else:
            persentase_tidak_depresi = 100 - persentase_depresi
            hasil_teks = f"Berdasarkan jawaban Anda, sistem memprediksi ada {round(persentase_tidak_depresi, 2)}% kemungkinan Anda tidak mengalami gejala depresi saat ini."

        # --- Logika Pesan Peringatan Dinamis ---

        # Periksa jawaban untuk pertanyaan bunuh diri
        if jawaban_user_dict.get('suicidal_thoughts') == '1':
            # Jika "Ya", gunakan pesan darurat
            disclaimer = (
                "PERINGATAN SERIUS: Jawaban Anda mengindikasikan adanya pemikiran untuk menyakiti diri sendiri. "
                "Anda tidak sendirian dan bantuan tersedia. Sangat disarankan untuk segera menghubungi layanan darurat kesehatan jiwa berikut:\n"
                "Kemenkes: (021) 500-454 atau 119 ext. 8 | "
                "LSM Jangan Bunuh Diri: (021) 9696 9293"
            )
        else:
            # Jika "Tidak", gunakan pesan standar
            disclaimer = (
                "Peringatan: Hasil ini bukanlah diagnosis medis. Ini adalah alat bantu deteksi dini. "
                "Sangat disarankan untuk berkonsultasi dengan psikolog atau profesional kesehatan mental."
            )
        
        # Kirim hasil dan disclaimer yang sesuai ke template
        return render_template('hasil.html', hasil_prediksi=hasil_teks, disclaimer=disclaimer)

    except Exception as e:
        return f"Terjadi error: {e}"

if __name__ == '__main__':
    app.run(debug=True)