#!/usr/bin/env python

from sklearn import datasets

# This example loads the IRIS dataset and classifies
# using our neural network implementation.
# The results are visualized in a 2D-plot.

def main():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

if __name__=='__main__':
    main()
