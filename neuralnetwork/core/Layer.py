import numpy as np

from neuralnetwork.activationfunctions.ActivationReLU import ActivationReLU
from neuralnetwork.activationfunctions.ActivationSigmoid import ActivationSigmoid
from neuralnetwork.activationfunctions.ActivationSoftmax import ActivationSoftmax
from neuralnetwork.activationfunctions.ActivationStep import ActivationStep
from neuralnetwork import settings
from neuralnetwork.activationxlossfunctions.ActivationSoftmaxLossCategoricalCrossentropy import ActivationSoftmaxLossCategoricalCrossentropy

np.random.seed(0)


class Layer:

    def __init__(self, nb_inputs, nb_neurons, activation_type):
        self.synaptic_weights = np.random.randn(nb_inputs, nb_neurons)/10  # reversed nb_neurons and nb_inputs to avoid using transpose
        self.activation_type = activation_type
        self.biases = np.zeros((1, nb_neurons))
        self.output = 0
        self.inputs = np.array([])
        self.dweights = np.array([])
        self.dbiases = np.array([])
        self.dinputs = np.array([])

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.synaptic_weights) + self.biases
        if self.activation_type == "Softmax":
            softmax = ActivationSoftmax()
            softmax.forward(self.output)
            self.output = softmax.output
        elif self.activation_type == "Step":
            step = ActivationStep()
            step.forward(self.output)
            self.output = step.output
        elif self.activation_type == "Sigmoid":
            sigmoid = ActivationSigmoid()
            sigmoid.forward(self.output)
            self.output = sigmoid.output
        elif self.activation_type == "ReLU":
            relu = ActivationReLU()
            relu.forward(self.output)
            self.output = relu.output

    def forward_last_layer_and_calculate_loss(self, inputs, y_true):
        self.inputs = inputs
        self.output = np.dot(inputs, self.synaptic_weights) + self.biases

        softmax_crossentropy = ActivationSoftmaxLossCategoricalCrossentropy()
        loss = softmax_crossentropy.forward(self.output, y_true)
        self.output = softmax_crossentropy.output

        softmax_crossentropy.backward(softmax_crossentropy.output, y_true)
        self.dinputs = softmax_crossentropy.dinputs
        return loss

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.synaptic_weights.T)

    def print_output(self):
        print(self.output[:settings.NB_LINES_PRINTED])
