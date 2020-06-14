import attr
import numpy as np  # type: ignore

from simple_neuralnet_python.sigmoid_tools import sigmoid, sigmoid_derivative


@attr.s(auto_attribs=True)
class NeuralNet():
    input_size: int
    output_size: int
    hidden_size: int = 4
    max_iterations: int = 100000
    learning_rate: float = 0.15
    weights_1: np.ndarray = attr.ib(init=False)
    weights_2: np.ndarray = attr.ib(init=False)

    def __attrs_post_init__(self):
        # Random weights between input and hidden layer
        self.weights_1 = 2 * np.random.random((self.input_size, self.hidden_size)) - 1
        # Weights between hidden and output layer
        self.weights_2 = 2 * np.random.random((self.hidden_size, self.output_size)) - 1

    def train(self, train_in: np.ndarray, train_out: np.ndarray):
        """Train the neural network using a training set.
        Arguments:
        train_in -- The training data inputs (array)
        train_out -- Training data expected outputs (array)
        """
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
                sigmoid_derivative(output_layer)
            hidden_error = output_adjustment.dot(self.weights_2.T)
            hidden_adjustment = hidden_error * self.learning_rate * \
                sigmoid_derivative(hidden_layer)
            # Actually adjust the weights
            self.weights_2 += hidden_layer.T.dot(output_adjustment)
            self.weights_1 += train_in.T.dot(hidden_adjustment)

    def classify(self, inputs: np.ndarray, training: bool = False):
        """Classify given data using the neural network.
        Arguments:
        inputs -- The input data (array)
        training -- If classify used for network training, also returns
          hidden layer (boolean)
        Returns:
        hidden_layer -- The hidden layer values (usually not needed)
        output_layer -- The classification results (array)
        """
        hidden_layer = sigmoid(np.dot(inputs, self.weights_1))
        output_layer = sigmoid(np.dot(hidden_layer, self.weights_2))
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
    nn = NeuralNet(input_size=3, output_size=1)
    # Train the network
    nn.train(train_in, train_out)
    # Test data
    test_in = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    # Classify
    results = nn.classify(test_in)
    print('Test data classification results (should be 1 0 1):')
    for result in results:
        print(int(np.rint(result[0])))


if __name__ == '__main__':
    main()
