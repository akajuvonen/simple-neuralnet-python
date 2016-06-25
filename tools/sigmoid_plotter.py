#!/usr/bin/env python

from neuralnet import NeuralNet
import matplotlib.pyplot as plt

def main():
    net = NeuralNet()

    sigmoid_values = []
    for i in range(-10,11):
        sigmoid_values.append(net._sigmoid(i))
    sigmoid_deriv_values = []
    for i in sigmoid_values:
        sigmoid_deriv_values.append(net._sigmoid_deriv(i))

    plt.plot(range(-10,11),sigmoid_values)
    plt.plot(range(-10,11),sigmoid_deriv_values)
    plt.ylabel('Value')
    plt.title('Sigmoid and its derivative')
    plt.grid(True)
    plt.savefig('sigmoid.png')
    plt.show()

if __name__=='__main__':
    main()
