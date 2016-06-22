import unittest
from neuralnet import NeuralNet

class NeuralNetTest(unittest.TestCase):
    def setUp(self):
        """Setup method for neural net tests"""
        self.net = NeuralNet()
