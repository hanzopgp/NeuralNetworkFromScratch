import numpy as np

from neuralnetwork.lossfunctions.Loss import Loss


class LossCategoricalCrossentropy(Loss):

    def __init__(self):
        self.dinputs = []

    def forward(self, y_pred, y_true):
        samples = len(y_pred)  # Number of sample in a batch
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Clipping data to prevent div by 0
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)  # Number of samples
        labels = len(dvalues[0])  # Number of labels in every sample
        if len(y_true.shape) == 1:  # If labels are sparse,
            y_true = np.eye(labels)[y_true]  # turn them into one-hot vector
        self.dinputs = -y_true / dvalues  # Calculate gradient
        self.dinputs = self.dinputs / samples  # Normalize gradient

