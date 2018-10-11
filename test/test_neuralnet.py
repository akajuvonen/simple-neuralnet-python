import unittest

import numpy as np

from simple_neuralnet_python import NeuralNet
from simple_neuralnet_python.neuralnet import NetworkNotTrainedException
from simple_neuralnet_python.neuralnet import TrainingNotSuccessfulException


class NeuralNetTest(unittest.TestCase):
    """Tests for NeuralNet class"""

    def setUp(self):
        """Setup method for neural net tests"""
        self.net = NeuralNet(max_iterations=50000)

    def testSigmoid(self):
        """Sigmoid method tests"""
        # Sigmoid function value should be close to 0
        i = -10
        result = self.net._sigmoid(i)
        self.assertTrue(result < 0.1)
        # Should be 0.5
        i = 0
        result = self.net._sigmoid(i)
        self.assertEqual(result, 0.5)
        # Should be close to 1
        i = 10
        result = self.net._sigmoid(i)
        self.assertTrue(result > 0.9)

    def testSigmoidDerivative(self):
        """Sigmoid deritative tests"""
        # Note that we must pass the values of the sigmoid function
        # to the sigmoid derivative, not the values of i themselves.
        # This is more useful for neural network implementation.
        #
        # Should be close to 0
        i = -10
        result = self.net._sigmoid_deriv(self.net._sigmoid(i))
        self.assertTrue(result < 0.1)
        # Should be 0.25
        i = 0
        result = self.net._sigmoid_deriv(self.net._sigmoid(i))
        self.assertEqual(result, 0.25)
        # Should be close to 0
        i = 10
        result = self.net._sigmoid_deriv(self.net._sigmoid(i))
        self.assertTrue(result < 0.1)

    def testTrainedFlag(self):
        """Test that the trained-flag works correctly"""
        # Should be false before training
        self.assertFalse(self.net.trained)
        # Train the net
        self.net.train(np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1], [
                       0, 1, 1]]), np.array([[1], [0], [1], [0]]))
        # Now should be True after training
        self.assertTrue(self.net.trained)

    def testNotTrainedException(self):
        """Test that NetworkNotTrained exception is correctly raised"""
        self.assertRaises(NetworkNotTrainedException,
                          self.net.classify, np.array([1]), np.array([1]))

    def testTrainingNotSuccessfulException(self):
        """Test that exception is raised when training does not succeed"""
        self.net.max_iterations = 1
        self.assertRaises(TrainingNotSuccessfulException, self.net.train,
                          np.array([[1, 0, 0], [0, 0, 1],
                                   [1, 0, 1], [0, 1, 1]]),
                          np.array([[1], [0], [1], [0]]))

    def testClassify(self):
        "Test that neural network classification works"
        train_in = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 1],
                             [1, 1, 0], [0, 1, 0], [0, 0, 0]])
        train_out = np.array([[1], [0], [1], [1], [0], [0]])
        # Init the network instance
        nn = NeuralNet()
        # Train the network
        nn.train(train_in, train_out)
        # Test data
        test_in = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        # Classify
        test_out = nn.classify(test_in)
        self.assertEqual(np.rint(test_out.item(0)).astype(int), 1)
        self.assertEqual(np.rint(test_out.item(1)).astype(int), 0)
        self.assertEqual(np.rint(test_out.item(2)).astype(int), 1)
