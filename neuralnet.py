#!/usr/bin/env python

import numpy as np

class NeuralNet():
    def __init__(self,hidden_size=4,max_iterations=100000,learning_rate=0.15):
        """The init method.
        Arguments:
        train_in -- Training set inputs (array)
        train_out -- Training set outputs (array)
        hidden_size -- How many neurons in hidden layer (int)
        iterations -- How many max iterations run in training (int)
        learning_rate -- Smaller LR means smaller jumps when learning
        """
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        # Is the network already trained
        self.trained = False

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

    def train(self,train_in,train_out):
        """Train the neural network using a training set."""
        # Init weights between -1 and 1
        # Weights between input and hidden layer
        # Notice that the shape tuple is inverted here using [::-1]
        self.weights_1 = 2*np.random.random((train_in.shape[1],self.hidden_size))-1
        # Weights between hidden and output layer
        self.weights_2 = 2*np.random.random((self.hidden_size,train_out.shape[1]))-1
        i=0
        for _ in range(self.max_iterations):
            # First, classify the training data using the network
            hidden_layer,output_layer = self.classify(train_in,True)
            # Calculate errors and adjustments
            output_error = train_out - output_layer
            # Print the error every 1000 iterations
            if i%1000==0:
                # Mean squared error
                mse = np.mean(np.power(output_error,2))
                print(mse)
            i+=1
            output_adjustment = output_error*self.learning_rate * self._sigmoid_deriv(output_layer)
            hidden_error = output_adjustment.dot(self.weights_2.T)
            hidden_adjustment = hidden_error*self.learning_rate * self._sigmoid_deriv(hidden_layer)
            # Actually adjust the weights
            self.weights_2 += hidden_layer.T.dot(output_adjustment)
            self.weights_1 += train_in.T.dot(hidden_adjustment)
        # Set the trained flag to True
        self.trained = True


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
        # Return also hidden layer for training
        if training:
            return hidden_layer,output_layer
        else:
            return output_layer

def main():
    # Training data inputs and outputs
    # Notice that the outputs correlate to the first element of each data point
    train_in = np.array([[1,0,0],[0,0,1],[1,1,1],[1,1,0],[0,1,0],[0,0,0]])
    train_out = np.array([[1],[0],[1],[1],[0],[0]])
    # Init the network instance
    nn = NeuralNet()
    # Train the network
    nn.train(train_in,train_out)
    # Test data
    test_in = np.array([[1,0,1],[0,1,0],[1,1,1]])
    # Classify
    train_out = nn.classify(test_in)
    # Should print approx. 1 0 1
    print('Test data classification results (should be 1 0 1):')
    print(np.rint(train_out).astype(int))

if __name__=='__main__':
    main()
