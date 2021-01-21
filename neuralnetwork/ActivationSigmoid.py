import numpy as np


class ActivationSigmoid:

    def __init__(self):
        self.output = []

    def forward(self, inputs):  # more challenging to compute
        self.output = 1 / (1 + np.exp(-inputs))