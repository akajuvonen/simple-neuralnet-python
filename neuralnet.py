#!/usr/bin/env python

import numpy as np

class NeuralNet():
    def __init__(self,train_in,train_out,hl_size,iterations=10000):
        """The init method.
        Arguments:
        train_in -- Training set inputs (array)
        train_out -- Training set outputs (array)
        hl_size -- Hidden layer size, e.g., (4,1) (tuple)
        iterations -- How many iterations run in training (int)
        """
        self.train_in = train_in
        self.train_out = train_out
        self.iterations = iterations
        # Init weights between -1 and 1
        # Weights between input and hidden layer
        # Notice that the shape tuple is inverted here using [::-1]
        self.weights_1 = 2*np.random.random(train_in.shape[::-1])-1
        # Weights between hidden and output layer
        self.weights_2 = 2*np.random.random(hl_size)-1
        # Train the network
        self.train()

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

    def train(self):
        """Train the neural network using a training set."""
        for _ in range(self.iterations):
            # First, classify the training data using the network
            hidden_layer,output_layer = self.classify(self.train_in,True)
            # Calculate errors and adjustments
            output_error = self.train_out - output_layer
            output_adjustment = output_error * self._sigmoid_deriv(output_layer)
            hidden_error = output_adjustment.dot(self.weights_2.T)
            hidden_adjustment = hidden_error * self._sigmoid_deriv(hidden_layer)
            # Actually adjust the weights
            self.weights_2 += hidden_layer.T.dot(output_adjustment)
            self.weights_1 += self.train_in.T.dot(hidden_adjustment)


    def classify(self,inputs,training=False):
        """Classify given data using the neural network.
        Arguments:
        inputs -- The input data (array)
        training -- If classify used for network training, also returns
          hidden layer (boolean)
        Returns:
        hidden_layer -- The hidden layer values (usually not needed)
        output_layer -- The classification results (array)
        """
        hidden_layer = self._sigmoid(np.dot(inputs,self.weights_1))
        output_layer = self._sigmoid(np.dot(hidden_layer,self.weights_2))
        if training:
            return hidden_layer,output_layer
        else:
            return output_layer

def main():
    train_in = np.array([[1,0,0],[0,0,1],[1,1,1],[1,1,0]])
    train_out = np.array([[1],[0],[1],[1]])
    nn = NeuralNet(train_in,train_out,(4,1))

if __name__=='__main__':
    main()
