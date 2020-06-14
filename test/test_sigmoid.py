from simple_neuralnet_python import sigmoid, sigmoid_derivative


def test_sigmoid():
    """Sigmoid function tests"""
    assert sigmoid(-10) < .1
    assert sigmoid(0) == .5
    assert sigmoid(10) > .9


def test_sigmoid_derivative():
    """Sigmoid deritative tests"""
    assert sigmoid_derivative(sigmoid(-10)) < .1
    assert sigmoid_derivative(sigmoid(0)) == .25
    assert sigmoid_derivative(sigmoid(10)) < .1
