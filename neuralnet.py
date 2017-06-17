#!/usr/bin/env python

import numpy as np


class NetworkNotTrainedException(Exception):
    """This exception is raised when neural network is used
    to classify data before it has been trained.
    """
    pass


class TrainingNotSuccessfulException(Exception):
    """Raised when the calculated MSE is too large after training."""
    pass


class NeuralNet():

    def __init__(self, hidden_size=4, max_iterations=100000,
                 learning_rate=0.15):
        """The init method.
        Arguments:
        hidden_size -- How many neurons in hidden layer (int)
        iterations -- How many max iterations run in training (int)
        learning_rate -- Smaller LR means smaller jumps when learning
        """
        self.hidden_size = hidden_size
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        # Is the network already trained
        self.trained = False
        # What is the minimum MSE that we want in order to consider
        # the training successful, i.e., if MSE is too large, the network
        # did not train very well using the current training data.
        # For now this limit is completely arbitrary.
        self.mse_limit = 0.01

    def _sigmoid(self, x):
        """The sigmoid function (or it's derivative).
        Arguments:
        x -- The weighted sum of an input
        Returns:
        The sigmoid function value.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, y):
        """The sigmoid derivative function.
        Arguments:
        y -- The neural network outputs. Notice that
        these values have already been calculated using
        the sigmoid function. An actual sigmoid derivative
        should be sigmoid(x)*(1-sigmoid(x)).
        Return:
        The sigmoid derivative.
        """
        return y * (1 - y)

    def train(self, train_in, train_out):
        """Train the neural network using a training set.
        Arguments:
        train_in -- The training data inputs (array)
        train_out -- Training data expected outputs (array)
        """
        # Set the trained flag to True
        self.trained = True
        # Init weights between -1 and 1
        # Weights between input and hidden layer
        self.weights_1 = 2 * \
            np.random.random((train_in.shape[1], self.hidden_size)) - 1
        # Weights between hidden and output layer
        self.weights_2 = 2 * \
            np.random.random((self.hidden_size, train_out.shape[1])) - 1
        i = 0
        for _ in range(self.max_iterations):
            # First, classify the training data using the network
            hidden_layer, output_layer = self.classify(train_in, True)
            # Calculate errors and adjustments
            output_error = train_out - output_layer
            # Print the error every 1000 iterations
            if i % 10000 == 0:
                # Mean squared error
                mse = np.mean(np.power(output_error, 2))
                print('MSE in iteration %d: %f' % (i, mse))
            i += 1
            output_adjustment = output_error * self.learning_rate * \
                self._sigmoid_deriv(output_layer)
            hidden_error = output_adjustment.dot(self.weights_2.T)
            hidden_adjustment = hidden_error * self.learning_rate * \
                self._sigmoid_deriv(hidden_layer)
            # Actually adjust the weights
            self.weights_2 += hidden_layer.T.dot(output_adjustment)
            self.weights_1 += train_in.T.dot(hidden_adjustment)
        # If training was not successful = MSE too large
        if mse > self.mse_limit:
            raise TrainingNotSuccessfulException('Mean squared error is too \
                    large, training was not successful')

    def classify(self, inputs, training=False):
        """Classify given data using the neural network.
        Arguments:
        inputs -- The input data (array)
        training -- If classify used for network training, also returns
          hidden layer (boolean)
        Returns:
        hidden_layer -- The hidden layer values (usually not needed)
        output_layer -- The classification results (array)
        """
        # If not yet trained
        if not self.trained:
            raise NetworkNotTrainedException('The network must be trained \
                    before classification')
        hidden_layer = self._sigmoid(np.dot(inputs, self.weights_1))
        output_layer = self._sigmoid(np.dot(hidden_layer, self.weights_2))
        # Return also hidden layer for training
        if training:
            return hidden_layer, output_layer
        else:
            return output_layer


def main():
    # Training data inputs and outputs
    # Notice that the outputs correlate to the first element of each data point
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
    # Should print approx. 1 0 1
    print('Test data classification results (should be 1 0 1):')
    result0 = np.rint(test_out.item(0)).astype(int)
    result1 = np.rint(test_out.item(1)).astype(int)
    result2 = np.rint(test_out.item(2)).astype(int)
    print("%d %d %d" % (result0, result1, result2))

if __name__ == '__main__':
    main()
