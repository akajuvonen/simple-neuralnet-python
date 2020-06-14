import numpy as np


def sigmoid(x: np.ndarray):
    """The sigmoid function (or it's derivative).
    Arguments:
    x -- The weighted sum of an input
    Returns:
    The sigmoid function value.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y: np.ndarray):
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
