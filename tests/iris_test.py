#!/usr/bin/env python

from sklearn import datasets
from random import shuffle

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
    # Split the shuffled indexes into half, to training and test
    # The int conversion is needed in python 3, I think (for odd number of indexes)
    train_idx = idx[:int(len(idx)/2)]
    test_idx = idx[int(len(X)/2):]

if __name__=='__main__':
    main()
