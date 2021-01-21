import numpy as np


class Loss:

    def __init__(self):
        self.output = []

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)  # Calculates sample losses
        data_loss = np.mean(sample_losses)  # Mean loss
        self.output = data_loss
