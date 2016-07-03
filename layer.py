#!/usr/bin/env python

from numpy.random import random

class Layer():
    """A neural network layer class that has it's own weights."""
    def __init__(self,neurons,inputs):
        """Init for layer.
        Arguments:
        neurons -- How many neurons in this layer (int)
        inputs -- How many inputs per neuron (int)
        """
        self.weights = random((inputs,neurons))

def main():
    # Create a layer and print its weights
    layer = Layer(5,3)
    print(layer.weights)

if __name__=='__main__':
    main()
