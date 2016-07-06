import unittest
from layer import Layer

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
        pass
