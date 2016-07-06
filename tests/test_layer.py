import unittest
from layer import Layer
from numpy import amax,amin

class NeuralNetTest(unittest.TestCase):
    """Tests for Layer class"""
    def setUp(self):
        """Setup for Layer tests"""
        # How many neurons in layer
        self.neurons = 5
        # How many inputs per neuron
        self.inputs = 3
        self.layer = Layer(self.neurons,self.inputs)

    def testLayerWeights(self):
        """Layer weight test"""
        # Make sure that the layer weights array is the right shape
        self.assertEqual(self.layer.weights.shape,(3,5))
        # Test that the min and max of weights are between -1 and 1
        self.assertTrue(amax(self.layer.weights)<=1.0)
        self.assertTrue(amin(self.layer.weights)>=-1.0)

