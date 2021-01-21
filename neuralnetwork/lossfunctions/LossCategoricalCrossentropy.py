import numpy as np

from neuralnetwork.lossfunctions.Loss import Loss


class LossCategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)  # Number of sample in a batch
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Clipping data to prevent div by 0
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)