#!/usr/bin/env python

from numpy import exp

class NeuralNet():
    def __init__(self):
        pass

    def _sigmoid(self,x):
        """The sigmoid function (or it's derivative).
        Arguments:
        x -- The weighted sum of an input
        Returns:
        The sigmoid function value.
        """
        return 1/(1+exp(-x))

    def _sigmoid_deriv(self,y):
        return y*(1-y)

    def train(self):
        pass

    def classify(self):
        pass

def main():
    nn = NeuralNet()

if __name__=='__main__':
    main()
