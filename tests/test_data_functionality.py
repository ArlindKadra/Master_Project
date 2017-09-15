import unittest
import numpy as np
from model import Input


class TestData(unittest.TestCase):

    def setUp(self):
        self.input = Input()

    def test_data_shape(self):
        data = self.input.read_data()
        for label, matrix in data:
            self.assertEqual(matrix.shape[0], 64)


    def test_get_example(self):
        matrix_a = np.zeros((64,30))
        matrix_b = np.zeros((64,100))
        matrix_c = np.zeros((64,50))
        window_data = [("data1", matrix_a), ("data2", matrix_b), ("data3", matrix_c)]
        self.input.getExample(window_data)
        
