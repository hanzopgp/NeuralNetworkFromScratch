import numpy as np


class Functions:

    @staticmethod
    def activation_ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def activation_softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # avoiding negative values et overflow errors
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)  # normalizing values

    @staticmethod
    def normalization_sigmoid(x):
        return 1 / (1 + np.exp(-x))
