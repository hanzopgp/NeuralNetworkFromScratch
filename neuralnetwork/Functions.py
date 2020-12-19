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
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
