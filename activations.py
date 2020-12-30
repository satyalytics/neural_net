""" 
Activation functions in neural network relu for regression and sigmoid for classification
"""
import numpy as np

def relu(X):
    return max(0, X)


def sigmoid(x):
    return 1/ (1+np.exp(-x))