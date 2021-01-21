import numpy as np

from neuralnetwork.ActivationReLU import ActivationReLU
from neuralnetwork.ActivationSigmoid import ActivationSigmoid
from neuralnetwork.ActivationSoftmax import ActivationSoftmax
from neuralnetwork.ActivationStep import ActivationStep

np.random.seed(0)


class Layer:

    def __init__(self, nb_inputs, nb_neurons, activation_type):
        self.synaptic_weights = np.random.randn(nb_inputs, nb_neurons)/10  # reversed nb_neurons and nb_inputs to avoid using transpose
        self.activation_type = activation_type
        self.biases = np.zeros((1, nb_neurons))
        self.output = 0

    def forward(self, inputs):
        self.output = np.dot(inputs, self.synaptic_weights) + self.biases
        if self.activation_type == "softmax":
            softmax = ActivationSoftmax()
            softmax.forward(self.output)
            self.output = softmax.output
        elif self.activation_type == "step":
            step = ActivationStep()
            step.forward(self.output)
            self.output = step.output
        elif self.activation_type == "sigmoid":
            sigmoid = ActivationSigmoid()
            sigmoid.forward(self.output)
            self.output = sigmoid.output
        elif self.activation_type == "ReLU":
            relu = ActivationReLU()
            relu.forward(self.output)
            self.output = relu.output

    def print_output(self):
        print(self.output)
