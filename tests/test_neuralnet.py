import unittest
from neuralnet import NeuralNet

class NeuralNetTest(unittest.TestCase):
    def setUp(self):
        """Setup method for neural net tests"""
        self.net = NeuralNet()

    def testSigmoid(self):
        """Sigmoid method tests"""
        # Sigmoid function value should be close to 0
        i=-10
        result = self.net._sigmoid(i)
        self.assertTrue(result<0.1)
        # Should be 0.5
        i=0
        result = self.net._sigmoid(i)
        self.assertEquals(result,0.5)
        # Should be close to 1
        i=10
        result = self.net._sigmoid(i)
        self.assertTrue(result>0.9)
