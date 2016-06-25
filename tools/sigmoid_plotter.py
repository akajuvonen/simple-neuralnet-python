#!/usr/bin/env python

from neuralnet import NeuralNet

def main():
    net = NeuralNet()
    sigmoid_values = []
    for i in range(-10,11):
        sigmoid_values.append(net._sigmoid(i))
    sigmoid_deriv_values = []
    for i in sigmoid_values:
        sigmoid_deriv_values.append(net._sigmoid_deriv(i))
    print(sigmoid_values)
    print(sigmoid_deriv_values)

if __name__=='__main__':
    main()
