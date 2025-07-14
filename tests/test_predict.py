import unittest
import numpy as np
import scipy.io as sio
import tensorflow as tf
from collections import Counter
import pandas as pd
import os

class TestPredictProcess(unittest.TestCase):
    def setUp(self):
        self.mat_path = "tests/DE_P1_I.mat"  # file data 3D (N, 4, 9, 9) atau (N, 9, 9, 4)
        self.model_paths = [
            "tests/modelfix_fold_1.h5",  # pastikan model valid ada di folder tests/
        ]
        self.output_excel = "tests/output_prediksi.xlsx"

    def test_predict_process_core(self):
        raw = sio.loadmat(self.mat_path)
        self.assertIn("data", raw, "Variabel 'data' tidak ditemukan di .mat")

        data_3d = raw["data"]
        self.assertEqual(data_3d.ndim, 4, "Data tidak berdimensi 4")

        # Format ke (N, 9, 9, 4)
        if data_3d.shape[-1] == 4:
            X = data_3d
        elif data_3d.shape[1] == 4:
            X = np.transpose(data_3d, (0, 2, 3, 1))
        else:
            self.fail(f"Dimensi channel tidak cocok: {data_3d.shape}")

        X = X.astype(np.float32)

        # Batas maksimal 10 detik
        if X.shape[0] > 10:
            X = X[:10]

        # Prediksi model
        all_preds = []
        for model_path in self.model_paths:
            self.assertTrue(os.path.exists(model_path), f"Model tidak ditemukan: {model_path}")
            model = tf.keras.models.load_model(model_path)
            preds = model.predict(X, verbose=0)
            pred_classes = np.argmax(preds, axis=1)
            all_preds.append(pred_classes)

        all_preds = np.array(all_preds)
        self.assertEqual(all_preds.shape[1], X.shape[0], "Jumlah prediksi tidak sesuai")

        # Voting
        voted = []
        for i in range(all_preds.shape[1]):
            detik_preds = all_preds[:, i]
            voted_class = Counter(detik_preds).most_common(1)[0][0]
            voted.append(voted_class)

        voted = np.array(voted)
        self.assertEqual(len(voted), X.shape[0], "Voting hasil tidak sesuai jumlah detik")

        # Simpan ke Excel
        df = pd.DataFrame({
            "Detik ke-": np.arange(1, len(voted)+1),
            "Prediksi Dominan": voted
        })
        for idx, pred in enumerate(all_preds):
            df[f"Model {idx+1}"] = pred

        df.to_excel(self.output_excel, index=False)
        self.assertTrue(os.path.exists(self.output_excel), "Excel hasil prediksi tidak disimpan")

if __name__ == '__main__':
    unittest.main()
