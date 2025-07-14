import unittest
import numpy as np
from sklearn.preprocessing import scale
from processing_3d import (
    read_file, get_vector_deviation, get_dataset_deviation,
    data_1Dto2D, preprocess_to_3D
)

class TestPreprocessing3D(unittest.TestCase):

    def setUp(self):
        # Simulasi data seperti yang ada di file .mat
        self.fake_mat = {
            "data": np.random.rand(10, 64),  # 10 detik, 64 fitur (tanpa baseline)
            "base_data": np.random.rand(2, 64),
            "data_seconds_list": np.array([[5, 5]]),  # 2 trial, masing-masing 5 detik
            "label_1": np.array([1]*10),
            "label_2": np.array([0]*10),
            "label_3": np.array([1]*10)
        }

    def test_read_file(self):
        td, bd, v, a, d, s = read_file(self.fake_mat)
        self.assertEqual(td.shape, (10, 64))
        self.assertEqual(bd.shape, (2, 64))
        self.assertEqual(len(v), 10)

    def test_vector_deviation(self):
        v1 = np.array([2, 4, 6])
        v2 = np.array([1, 2, 3])
        result = get_vector_deviation(v1, v2)
        expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_dataset_deviation_shape(self):
        trial = np.random.rand(10, 64)
        base = np.random.rand(2, 64)
        seconds = np.array([[5, 5]])
        result = get_dataset_deviation(trial, base, seconds)
        self.assertEqual(result.shape, (10, 64))

    def test_data_1Dto2D_shape(self):
        vector = np.arange(16)
        matrix = data_1Dto2D(vector)
        self.assertEqual(matrix.shape, (9, 9))
        self.assertTrue(isinstance(matrix, np.ndarray))

    def test_preprocess_to_3D_output(self):
        result = preprocess_to_3D(self.fake_mat, use_baseline="yes")
        self.assertIn("data", result)
        self.assertIn("label_1", result)
        self.assertEqual(result["data"].shape[1:], (4, 9, 9))  # shape = (N, 4, 9, 9)
        self.assertEqual(len(result["label_1"]), result["data"].shape[0])

if __name__ == "__main__":
    unittest.main()
