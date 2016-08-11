import unittest
from neuralnet import NeuralNet
import numpy as np

class NeuralNetTest(unittest.TestCase):
    """Tests for NeuralNet class"""
    def setUp(self):
        """Setup method for neural net tests"""
        self.net = NeuralNet(max_iterations=10000)

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

    def testSigmoidDerivative(self):
        """Sigmoid deritative tests"""
        # Note that we must pass the values of the sigmoid function
        # to the sigmoid derivative, not the values of i themselves.
        # This is more useful for neural network implementation.
        #
        # Should be close to 0
        i=-10
        result = self.net._sigmoid_deriv(self.net._sigmoid(i))
        self.assertTrue(result<0.1)
        # Should be 0.25
        i=0
        result = self.net._sigmoid_deriv(self.net._sigmoid(i))
        self.assertEquals(result,0.25)
        # Should be close to 0
        i=10
        result = self.net._sigmoid_deriv(self.net._sigmoid(i))
        self.assertTrue(result<0.1)

    def testTrainedFlag(self):
        """Test that the trained-flag works correctly"""
        # Should be false before training
        self.assertFalse(self.net.trained)
        # Train the net
        self.net.train(np.array([[1,0,0],[0,0,1],[1,0,1],[0,1,1]]),np.array([[1],[0],[1],[0]]))
        # Now should be True after training
        self.assertTrue(self.net.trained)
