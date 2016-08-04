#!/usr/bin/env python

from sklearn import datasets

def main():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

if __name__=='__main__':
    main()
