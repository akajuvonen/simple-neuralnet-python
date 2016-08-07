#!/usr/bin/env python

from sklearn import datasets
from random import shuffle
import numpy as np
from neuralnet import NeuralNet

# This example loads the IRIS dataset and classifies
# using our neural network implementation.
# The results are visualized in a 2D-plot.

def main():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    # Randomize (shuffle) the indexes
    # Shuffle randomizes idx in place
    idx = range(len(X))
    shuffle(idx)
    # Split the shuffled indexes into training and test
    train_idx = idx[:99]
    test_idx = idx[100:]
    # Initialize zero matrix for outputs in binary form
    Y_bin = np.zeros((len(Y),3),dtype=np.int)
    # Convert output from int to binary representation for neural network
    for i in range(len(Y)):
        Y_bin[i][Y[i]] = 1
    # Init and train the neural network
    net = NeuralNet(X[train_idx],Y_bin[train_idx])
    # Classify
    results = net.classify(X[test_idx])
    print(np.rint(results).astype(int))

if __name__=='__main__':
    main()
