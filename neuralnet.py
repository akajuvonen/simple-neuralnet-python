#!/usr/bin/env python

from numpy import exp

class NeuralNet():
    def __init__(self):
        pass

    def _sigmoid(self,x,deriv=False):
        """The sigmoid function (or it's derivative).
        Arguments:
        x -- The weighted sum of an input
        deriv -- Boolean, if True, we return the sigmoid derivative
        Returns:
        The sigmoid function value, or the derivative if needed.
        """
        if deriv:
            # Calculate the sigmoid derivative.
            # We use recursion because it looks better and
            # makes the code simple. There are faster ways of
            # doing this.
            return self._sigmoid(x)*(1-self._sigmoid(x))
        else:
            # Calculate the sigmoid
            return 1/(1+exp(-x))

    def train(self):
        pass

    def classify(self):
        pass

def main():
    nn = NeuralNet()

if __name__=='__main__':
    main()
