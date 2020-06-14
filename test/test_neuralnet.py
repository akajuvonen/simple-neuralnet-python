import numpy as np

from simple_neuralnet_python import NeuralNet


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
