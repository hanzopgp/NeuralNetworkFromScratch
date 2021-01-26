import numpy as np


class ActivationReLU:

    def __init__(self):
        self.output = np.array([])
        self.inputs = np.array([])
        self.dinputs = np.array([])

    def forward(self, inputs):  # best one most of the time
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
