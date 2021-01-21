import numpy as np


class Loss:

    def calculate(self, output, correct_output):
        sample_losses = self.forward(output, correct_output)  # Calculates sample losses
        data_loss = np.mean(sample_losses)  # Mean loss
        return data_loss
