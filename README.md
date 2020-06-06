# simple-neuralnet-python

A simple MLP neural network using Python.

## Installation

It's recommended to install the package inside a clean virtualenv.
The package has been tested with Python versions `3.6` and `3.8`.

You can install the package using pip:

```
pip install .
```

## Tests

Tests can be run with `python setup.py test`.

## Usage

### Using command line

You can run a basic example with command `neuralnet-run`, which will
run a basic sanity check using simple example data. It should print
something similar to the following:

```
MSE in iteration 0: 0.254802
MSE in iteration 10000: 0.000473
MSE in iteration 20000: 0.000214
MSE in iteration 30000: 0.000137
MSE in iteration 40000: 0.000100
MSE in iteration 50000: 0.000079
MSE in iteration 60000: 0.000065
MSE in iteration 70000: 0.000055
MSE in iteration 80000: 0.000048
MSE in iteration 90000: 0.000042
Test data classification results (should be 1 0 1):
1 0 1
```

### Importing module

You can import the neural net as a module and use it like this example:

```python
from simple_neuralnet_python import NeuralNet


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
```

## Utils

Package also includes a tool to visualize the used sigmoid function and its derivative.
You can run it by invoking `neuralnet-sigmoid-plotter`.
