import numpy as np


class Functions:

    @staticmethod
    def activation_ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def activation_step(x):
        if x > 0:
            return x
        else:
            return 0

    @staticmethod
    def activation_softmax(x):
        return np.exp(x)

    @staticmethod
    def normalization_sigmoid(x):
        return 1 / (1 + np.exp(-x))
