import numpy as np


class ActivationReLU:

    def __init__(self):
        self.output = []

    def forward(self, inputs):  # best one most of the time
        self.output = np.maximum(0, inputs)
