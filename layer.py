#!/usr/bin/env python

from numpy.random import random

class Layer():
    def __init__(self,neurons,inputs):
        self.weights = random((inputs,neurons))

def main():
    layer = Layer(5,3)
    print(layer.weights)

if __name__=='__main__':
    main()
