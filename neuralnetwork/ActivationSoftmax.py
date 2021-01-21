import numpy as np


class ActivationSoftmax:

    def __init__(self):
        self.output = []

    def forward(self, inputs):  # used in output layers to display probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # avoiding negative values et overflow errors
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # normalizing values
