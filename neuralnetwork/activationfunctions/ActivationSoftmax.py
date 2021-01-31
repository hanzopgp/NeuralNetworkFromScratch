import numpy as np


class ActivationSoftmax:

    def __init__(self):
        self.output = []
        self.dinputs = np.array([])

    def forward(self, inputs):  # used in output layers to display probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # avoiding negative values et overflow errors
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # normalizing values

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)  # init array
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):  # enumerate outputs and gradients
            single_output = single_output.reshape(-1, 1)  # flatten output array
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
