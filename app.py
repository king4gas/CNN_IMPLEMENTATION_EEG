from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
from processing_1d import decompose, get_labels, process_1d_from_mat
from processing_3d import preprocess_to_3D
from flask import Flask, render_template, request, redirect, url_for, send_file
from sklearn.metrics import precision_score, recall_score, f1_score
from training_cnn import train_model
import tensorflow as tf
import numpy as np
import pandas as pd
import shutil
import tempfile
import os
import scipy.io as sio
import zipfile
from training_cnn import get_training_results
from collections import Counter


app = Flask(__name__)

app.secret_key = "agas123"

# Folder konfigurasi
MODEL_FOLDER = os.path.join(os.getcwd(), 'static', 'downloads')
os.makedirs(MODEL_FOLDER, exist_ok=True)
UPLOAD_FOLDER = 'uploads/'
MODEL_FOLDER = 'models/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Inisialisasi progress
with open("progress.txt", "w") as f:
    f.write("0")


ALLOWED_EXTENSIONS = {'mat'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/progress')
def progress():
    try:
        with open("progress.txt", "r") as f:
            percent = int(f.read().strip())
    except:
        percent = 0
    return jsonify(progress=percent)

@app.route('/')
def index():
    return render_template('index.html')

#FITUR PREPROCESS 1D
import zipfile
from werkzeug.utils import secure_filename

@app.route("/preprocess_1d", methods=["GET", "POST"])
def preprocess_1d():
    if request.method == "POST":
        files = request.files.getlist("files")
        if not files or not all(allowed_file(f.filename) for f in files):
            return render_template("convert.html", error="Please upload valid .mat files.")

        output_paths = []
        try:
            for file in files:
                filename = secure_filename(file.filename)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
                    file.save(tmp.name)
                    temp_path = tmp.name

                mat_data = sio.loadmat(temp_path)
                result = process_1d_from_mat(mat_data)

                processed_filename = f"processed_1d_{filename}"
                save_path = os.path.join(MODEL_FOLDER, processed_filename)
                sio.savemat(save_path, result)
                output_paths.append(save_path)

            zip_filename = "hasil_1d_preprocessed.zip"
            zip_path = os.path.join(MODEL_FOLDER, zip_filename)
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for path in output_paths:
                    zipf.write(path, os.path.basename(path))

            return render_template("convert.html", download_url_1d=url_for("download_file", filename=zip_filename))

        except Exception:
            return render_template("convert.html", error="Error during processing. Please check your files.")

    return render_template("convert.html")

@app.route("/preprocess_3d", methods=["GET", "POST"])
def preprocess_3d():
    if request.method == "POST":
        files = request.files.getlist("files")
        if not files or not all(allowed_file(f.filename) for f in files):
            return render_template("convert.html", error="Please upload valid .mat files.")

        output_paths = []
        try:
            for file in files:
                filename = secure_filename(file.filename)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
                    file.save(tmp.name)
                    temp_path = tmp.name

                raw = sio.loadmat(temp_path)
                result = preprocess_to_3D(raw, use_baseline="yes")

                processed_filename = f"processed_3d_{filename}"
                save_path = os.path.join(MODEL_FOLDER, processed_filename)
                sio.savemat(save_path, result)
                output_paths.append(save_path)

            zip_filename = "hasil_3d_preprocessed.zip"
            zip_path = os.path.join(MODEL_FOLDER, zip_filename)
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for path in output_paths:
                    zipf.write(path, os.path.basename(path))

            return render_template("convert.html", download_url_3d=url_for("download_file", filename=zip_filename))

        except Exception:
            return render_template("convert.html", error="Error during processing. Please check your files.")

    return render_template("convert.html")

# FITUR TRAINING
@app.route('/train')
def upload_page():
    return render_template('training.html')

import threading
@app.route('/train', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Inisialisasi progress
        with open("progress.txt", "w") as f:
            f.write("0")

        # Jalankan training di background (non-blocking)
        thread = threading.Thread(target=train_model, args=(filepath, False))
        thread.start()

        # Redirect ke halaman progress
        return redirect(url_for('progress'))
    
    return 'Invalid file', 400

# FITUR VIEW RESULT TRAINING
@app.route('/train_result')
def train_result():
    model_files, results_df, averages = get_training_results()
    return render_template('train_result.html',
                           model_files=model_files,
                           metrics_df=results_df.to_dict(orient='records'),
                           averages=averages)

# FITUR PREDIKSI
def allowed_mat(filename: str) -> bool:
    return filename.lower().endswith(".mat")

CLASS_MAP = {
    0: ("Aksara A", "Aksara A.png"),
    1: ("Aksara I", "Aksara I.png"),
    2: ("Aksara U", "Aksara U.png"),
    3: ("Aksara Ē", "Aksara EE.png"),
    4: ("Aksara E", "Aksara E.png"),
    5: ("Aksara O", "Aksara O.png"),
}

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        result = session.pop('result', None)
        error = session.pop('error', None)
        return render_template("predict.html", result=result, error=error)

    # POST - submit file
    mat_file = request.files.get("file")
    model_files = request.files.getlist("models")

    if not (mat_file and model_files and allowed_mat(mat_file.filename)):
        session['error'] = "File tidak valid atau format salah."
        return redirect(url_for("predict"))

    # Simpan file .mat
    mat_path = os.path.join(UPLOAD_FOLDER, "input.mat")
    mat_file.save(mat_path)

    # Simpan semua model .h5
    saved_model_paths = []
    for i, file in enumerate(model_files):
        if file and file.filename.endswith(".h5"):
            path = os.path.join(MODEL_FOLDER, f"model_{i}.h5")
            file.save(path)
            saved_model_paths.append(path)

    if len(saved_model_paths) == 0:
        session['error'] = "Tidak ada file .h5 yang valid."
        return redirect(url_for("predict"))

    try:
        # Load data EEG
        raw = sio.loadmat(mat_path)
        if "data" not in raw:
            session['error'] = "Variabel 'data' tidak ditemukan dalam file .mat"
            return redirect(url_for("predict"))

        data_3d = raw["data"]
        if data_3d.ndim != 4:
            session['error'] = f"Data harus berdimensi 4, tetapi ditemukan: {data_3d.ndim}"
            return redirect(url_for("predict"))

        # Format ke (N, 9, 9, 4)
        if data_3d.shape[-1] == 4:
            X = data_3d
        elif data_3d.shape[1] == 4:
            X = np.transpose(data_3d, (0, 2, 3, 1))
        else:
            session['error'] = f"Dimensi channel tidak cocok: {data_3d.shape}"
            return redirect(url_for("predict"))

        X = X.astype(np.float32)

        # Batas 10 detik maksimal
        if X.shape[0] > 10:
            X = X[:10]

        # Prediksi semua model
        all_preds = []
        for model_path in saved_model_paths:
            model = tf.keras.models.load_model(model_path)
            preds = model.predict(X, verbose=0)
            pred_classes = np.argmax(preds, axis=1)
            all_preds.append(pred_classes)

        all_preds = np.array(all_preds)  # shape: (num_models, num_detik)

        # Voting per detik
        voted = []
        for i in range(all_preds.shape[1]):
            detik_preds = all_preds[:, i]
            voted_class = Counter(detik_preds).most_common(1)[0][0]
            voted.append(voted_class)

        voted = np.array(voted)

        # ===============================
        # 🆕 Mapping label angka ke teks:
        # ===============================
        mapped_voted = [CLASS_MAP.get(label, ("Unknown",))[0] for label in voted]

        df = pd.DataFrame({
            "Detik ke-": np.arange(1, len(voted) + 1),
            "Prediksi Dominan": mapped_voted
        })

        # Tambahkan prediksi tiap model (dalam bentuk teks)
        for idx, pred in enumerate(all_preds):
            mapped_pred = [CLASS_MAP.get(p, ("Unknown",))[0] for p in pred]
            df[f"Model {idx+1}"] = mapped_pred

        # Simpan ke Excel
        excel_path = os.path.join(UPLOAD_FOLDER, "hasil_prediksi.xlsx")
        df.to_excel(excel_path, index=False)

        # Ambil label global dominan
        most_common_class, freq = Counter(voted).most_common(1)[0]
        label_text, image_filename = CLASS_MAP.get(most_common_class, ("Unknown", "default.png"))

        result = {
            "text": f"Predicted Result: {label_text}",
            "img": url_for("static", filename=f"img/{image_filename}"),
            "download_link": url_for("download_prediction"),
            "info": f"{len(saved_model_paths)} model digunakan (voting per detik)."
        }

        session['result'] = result
        return redirect(url_for("predict"))

    except Exception as e:
        session['error'] = f"Terjadi kesalahan saat prediksi: {str(e)}"
        return redirect(url_for("predict"))

@app.route("/download_prediction")
def download_prediction():
    file_path = os.path.join(UPLOAD_FOLDER, "hasil_prediksi.xlsx")
    return send_file(file_path, as_attachment=True)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(MODEL_FOLDER, filename, as_attachment=True)

@app.route('/download_all')
def download_all_models():
    return send_from_directory(MODEL_FOLDER, "all_models_and_metrics.rar")
    
if __name__ == '__main__':
    app.run(debug=True)