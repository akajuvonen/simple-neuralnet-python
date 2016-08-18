# simple-neuralnet-python

A simple neural network using Python. Made mostly as an experiment and practice.

## Usage

Currently just run `python neuralnet.py` The networks trains using some training data, then classifies test data. On the command line the testing data outputs are printed. The data used can be seen from `main()`.

To use the network as a library, import it. Then create a class instance, you can see the available init parameters from the code comments. With simplest option: `net = NeuralNet()`, this is with default params. After that you need to train the network: `net.train(inputs,outputs)`. If the training is not successful (mean squared error too large), there will an exception (I know, this is a very crude way to do this but enough for this simple network). You can classify new data using `net.classify(test_data_inputs)`. If you try to classify before the model has been trained, an exception will be raised.

## Unit tests
The simplest way to run tests is running `nosetests -v` in the main directory. Nose must be installed for this to work. If the tests (this or Iris data test) cannot find the file `neuralnet.py`, you may need to export the neural net main folder to `PYTHONPATH` (on Linux).

## Iris data test
In the folder `tests/` there is a file `iris_test.py`. Run this using `python iris_test.py`. It will test the network using Iris dataset and print the percentage of correct classifications.
