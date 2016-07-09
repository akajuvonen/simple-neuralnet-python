#!/usr/bin/env python

import numpy as np

class NeuralNet():
    def __init__(self,train_in,train_out,iterations=10000):
        """The init method.
        Arguments:
        train_in -- Training set inputs (array)
        train_out -- Training set outputs (array)
        """
        self.train_in = train_in
        self.train_out = train_out
        self.iterations = iterations

    def _sigmoid(self,x):
        """The sigmoid function (or it's derivative).
        Arguments:
        x -- The weighted sum of an input
        Returns:
        The sigmoid function value.
        """
        return 1/(1+np.exp(-x))

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

    def train(self,inputs,outputs,iterations):
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
    train_in = np.array([[1,0,0],[0,0,1],[1,1,1],[1,1,0]])
    train_out = np.array([[1],[0],[1],[1]])
    nn = NeuralNet(train_in,train_out)

if __name__=='__main__':
    main()
