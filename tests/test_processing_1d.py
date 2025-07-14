import unittest
import numpy as np
import scipy.io as sio
from processing_1d import process_1d_from_mat, compute_DE, normalization, butter_bandpass_filter

class TestProcess1D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mat_data = sio.loadmat("tests/P1_Trial2_CS.mat")
        cls.result = process_1d_from_mat(cls.mat_data)

    def test_output_keys(self):
        keys = ["base_data", "data", "label_1", "label_2", "label_3", "data_seconds_list"]
        for key in keys:
            self.assertIn(key, self.result)

    def test_base_data_shape(self):
        self.assertEqual(self.result["base_data"].shape[1], 64)

    def test_data_shape(self):
        self.assertEqual(self.result["data"].shape[1], 64)

    def test_labels_length(self):
        data_len = self.result["data"].shape[0]
        self.assertEqual(len(self.result["label_1"]), data_len)
        self.assertEqual(len(self.result["label_2"]), data_len)
        self.assertEqual(len(self.result["label_3"]), data_len)

    def test_compute_de_output(self):
        dummy = np.random.randn(100)
        self.assertIsInstance(compute_DE(dummy), float)

    def test_bandpass_filter_length(self):
        dummy = np.random.randn(500)
        filtered = butter_bandpass_filter(dummy, 4, 8, fs=100)
        self.assertEqual(len(filtered), len(dummy))

    def test_normalization_output(self):
        dummy = np.random.randn(500)
        normed = normalization(dummy)
        self.assertAlmostEqual(normed.mean(), 0, places=1)
        self.assertAlmostEqual(normed.std(), 1, places=1)

if __name__ == '__main__':
    unittest.main()
