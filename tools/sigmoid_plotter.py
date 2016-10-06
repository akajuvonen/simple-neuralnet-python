#!/usr/bin/env python

from neuralnet import NeuralNet
import matplotlib.pyplot as plt

# Plots sigmoid function and it's derivative using matplotlib


def main():
    # Create the net that includes sigmoid functions
    net = NeuralNet()

    # Populate lists with sigmoid values
    sigmoid_values = []
    for i in range(-10, 11):
        sigmoid_values.append(net._sigmoid(i))
    sigmoid_deriv_values = []
    for i in sigmoid_values:
        sigmoid_deriv_values.append(net._sigmoid_deriv(i))

    # Plot and set some parameters for plt
    plt.plot(range(-10, 11), sigmoid_values)
    plt.plot(range(-10, 11), sigmoid_deriv_values)
    plt.ylabel('Value')
    plt.title('Sigmoid and its derivative')
    plt.grid(True)
    # Save fig and show
    plt.savefig('sigmoid.png')
    plt.show()

if __name__ == '__main__':
    main()
