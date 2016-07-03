#!/usr/bin/env python

from numpy import exp
from layer import Layer

class NeuralNet():
    def __init__(self):
        """The init method."""
        # A list of Layer-objects
        layers = []

    def _sigmoid(self,x):
        """The sigmoid function (or it's derivative).
        Arguments:
        x -- The weighted sum of an input
        Returns:
        The sigmoid function value.
        """
        return 1/(1+exp(-x))

    def _sigmoid_deriv(self,y):
        """The sigmoid derivative function.
        Arguments:
        y -- The neural network outputs. Notice that
        these values have already been calculated using
        the sigmoid function. An actual sigmoid derivative
        should be sigmoid(x)*(1-sigmoid(x)).
        Return:
        The sigmoid derivative.
        """
        return y*(1-y)

    def train(self,inputs,outputs,iterations=10000):
        """Train the neural network using a training set.
        Arguments:
        inputs -- The input data from a training set (array)
        outputs -- The correct outputs from the training set (array)
        iterations -- How many iterations to train the network (int, default=10000)
        """
        pass

    def classify(self):
        pass

def main():
    nn = NeuralNet()

if __name__=='__main__':
    main()
