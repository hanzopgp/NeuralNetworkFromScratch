import numpy as np


class Functions:

    @staticmethod
    def Activation_ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))