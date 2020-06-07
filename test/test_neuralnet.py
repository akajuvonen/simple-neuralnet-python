import unittest

import numpy as np

from simple_neuralnet_python import NeuralNet


def test_sigmoid():
    """Sigmoid function tests"""
    # Sigmoid function value should be close to 0
    net = NeuralNet(3, 1)
    i = -10
    result = net._sigmoid(i)
    assert result < .1
    # Should be 0.5
    i = 0
    result = net._sigmoid(i)
    assert result == .5
    # Should be close to 1
    i = 10
    result = net._sigmoid(i)
    assert result > .9

def test_sigmoid_derivative():
    """Sigmoid deritative tests"""
    net = NeuralNet(1, 3)
    # Should be close to 0
    i = -10
    result = net._sigmoid_deriv(net._sigmoid(i))
    assert result < .1
    # Should be 0.25
    i = 0
    result = net._sigmoid_deriv(net._sigmoid(i))
    assert result == .25
    # Should be close to 0
    i = 10
    result = net._sigmoid_deriv(net._sigmoid(i))
    assert result < .1

def test_classify():
    "Test that neural network classification works"
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
    test_out = nn.classify(test_in)
    expected = [1, 0, 1]
    for result, expected_result in zip(test_out, expected):
        assert int(np.rint(result)) == expected_result
