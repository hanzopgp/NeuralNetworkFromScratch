import numpy as np


class Functions:

    @staticmethod
    def activation_step(x):  # outdated most of the time because it makes regression harder
        res = []
        if x > 0:
            res.append(1)
        else:
            res.append(0)
        return res

    @staticmethod
    def activation_sigmoid(x):  # more challenging to compute
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activation_ReLU(x):  # best one most of the time
        return np.maximum(0, x)

    @staticmethod
    def activation_softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # avoiding negative values et overflow errors
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)  # normalizing values

