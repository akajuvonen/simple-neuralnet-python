# simple-neuralnet-python

A simple MLP neural network using Python. Made mostly as an experiment and practice.

## Installation

You need to have python, virtualenv and pip installed.

First of all, make sure you are using virtualenv, you will do yourself a big favor. You can create a new environment with `virtualenv env-name`, and activate it by running `source env-name/bin/activate`. After this, follow the instructions below. You can deactivate with command `deactivate` and return to the system python environment.

There is a makefile to make dependency installation easier. Just run `make init` inside of your python virtual environment and it will pull the necessary and tested dependencies using pip. The required packages are listed in `requirements.txt`.

## Usage

Run the network from command line using sample data with `make run `.

To use the network as a library, import it (currently no installation, so just make sure the file is available somewhere). Then create a class instance, you can see the available init parameters from the code comments. With simplest option: `net = NeuralNet()`, this is with default params. After that you need to train the network: `net.train(inputs,outputs)`. If the training is not successful (mean squared error too large), there will an exception (I know, this is a very crude way to do this but enough for this simple network). You can classify new data using `net.classify(test_data_inputs)`. If you try to classify before the model has been trained, an exception will be raised.

## Tests

Run `make test` to run unit tests with nose.

If you want to try the proper test with Iris dataset, run `make iris`. It will train and classify using the dataset and print out classification accuracy (should be between 90-100%).

## Cleaning

Clean the `*.pyc` files using `make clean`.
