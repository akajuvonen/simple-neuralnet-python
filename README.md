# simple-neuralnet-python

A simple neural network using Python. Made mostly as an experiment and practice.

## Usage

Currently just run `python neuralnet.py` The networks trains using some training data, then classifies test data. On the command line the testing data outputs are printed. The data used can be seen from `main()`.

## Unit tests
The simplest way to run tests is running `nosetests -v` in the main directory. Nose must be installed for this to work.

## Iris data test
In the folder `tests/` there is a file `test_iris.py`. Run this using `python test_iris.py`. It will test the network using Iris dataset and print the percentage of correct classifications.
