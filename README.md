# simple-neuralnet-python

A simple MLP neural network using Python. Made mostly as an experiment and practice.

## Installation

You need to have python, virtualenv and pip installed.

There is a makefile to make dependency installation easier. Just run `make init` or `make` and it will pull the necessary and tested dependencies using pip. The required packages are listed in `requirements.txt`. All the scripts use the virtual environment automatically when needed.

## Usage

Run the network from command line using sample data with `make run `. The network will run a simple test and training data example, where it attempts to classify binary values. It also prints the MSE (Mean Squared Error) every 10,000 iterations, so you can see how the training progresses. The expected output will be like the following:
```
MSE in iteration 0: 0.254802
MSE in iteration 10000: 0.000473
MSE in iteration 20000: 0.000214
MSE in iteration 30000: 0.000137
MSE in iteration 40000: 0.000100
MSE in iteration 50000: 0.000079
MSE in iteration 60000: 0.000065
MSE in iteration 70000: 0.000055
MSE in iteration 80000: 0.000048
MSE in iteration 90000: 0.000042
Test data classification results (should be 1 0 1):
1 0 1
```

To use the network as a library, import it (currently no installation, so just make sure the file is available somewhere). Then create a class instance, you can see the available init parameters from the code comments. With simplest option: `net = NeuralNet()`, this is with default params. After that you need to train the network: `net.train(inputs,outputs)`. If the training is not successful (mean squared error too large), there will an exception (I know, this is a very crude way to do this but enough for this simple network). You can classify new data using `net.classify(test_data_inputs)`. If you try to classify before the model has been trained, an exception will be raised.

## Tests

Run `make test` to run unit tests with nose.

If you want to try the proper test with Iris dataset, run `make iris`. It will train and classify using the dataset and print out classification accuracy (should be between 90-100%).

## Plotting the sigmoid function

For visualizing how the sigmoid function works, you can run `make plot` the draw a figure of the function. Note that Python Tkinter is needed for this, and it cannot be installed using pip. Therefore, virtualenv cannot install this. In Ubuntu, installing the package `python-tk` accomplishes this. It might be necessary to copy or symlink tcl and tk directories inside `.env/lib`.

## Cleaning

Clean the `*.pyc` files and the virtual environment using `make clean`.
