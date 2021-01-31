import numpy as np

from neuralnetwork.activationfunctions.ActivationSoftmax import ActivationSoftmax
from neuralnetwork.lossfunctions.LossCategoricalCrossentropy import LossCategoricalCrossentropy


class ActivationSoftmaxLossCategoricalCrossentropy:

    def __init__(self):
        self.activation_function = ActivationSoftmax()
        self.loss_function = LossCategoricalCrossentropy()
        self.output = []
        self.dinputs = []

    def forward(self, inputs, y_true):
        self.activation_function.forward(inputs)
        self.output = self.activation_function.output
        return self.loss_function.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):  # Faster backward step
        samples = len(dvalues)
        if len(y_true.shape) == 2:  # If labels are one-hot encoded,
            y_true = np.argmax(y_true, axis=1)  # turn them into discrete values
        self.dinputs = dvalues.copy()  # Copy so we can safely modify
        self.dinputs[range(samples), y_true] -= 1  # Calculate gradient
        self.dinputs = self.dinputs / samples  # Normalize gradient



