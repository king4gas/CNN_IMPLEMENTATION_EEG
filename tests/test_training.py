import unittest
import os
import scipy.io as sio
from training_cnn import train_model

class TestTrainingProcess(unittest.TestCase):

    def setUp(self):
        self.test_mat_path = "tests/3D.mat"
        assert os.path.exists(self.test_mat_path), "File .mat tidak ditemukan."

    def test_train_model_without_html(self):
        result = train_model(self.test_mat_path, return_html=False)
        
        # Cek struktur output
        self.assertIn("model_files", result)
        self.assertIn("metrics_df", result)
        self.assertIn("averages", result)

        # Cek model tersimpan
        self.assertTrue(len(result["model_files"]) > 0)
        for fname in result["model_files"]:
            self.assertTrue(fname.endswith(".h5"))
            self.assertTrue(os.path.exists(os.path.join("static/models/", fname)))

        # Cek hasil evaluasi
        avg = result["averages"]
        self.assertGreaterEqual(avg["Accuracy"], 0)
        self.assertLessEqual(avg["Accuracy"], 100)

    def tearDown(self):
        # Optional: hapus model hasil training jika perlu
        pass

if __name__ == "__main__":
    unittest.main()
