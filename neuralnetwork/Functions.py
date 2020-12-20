import numpy as np


class Functions:

    @staticmethod
    def activation_ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def activation_softmax(x):
        exp_values = np.exp(x)
        return exp_values / np.sum(exp_values)

    @staticmethod
    def normalization_sigmoid(x):
        return 1 / (1 + np.exp(-x))
