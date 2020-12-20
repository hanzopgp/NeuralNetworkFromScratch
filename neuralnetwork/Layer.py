import numpy as np

from neuralnetwork.Functions import Functions

np.random.seed(0)


class Layer:

    def __init__(self, nb_inputs, nb_neurons):
        self.synaptic_weights = np.random.randn(nb_inputs, nb_neurons)/10  # reversed nb_neurons and nb_inputs to avoid using transpose
        self.biases = np.zeros((1, nb_neurons))
        self.output = 0

    def forward(self, inputs):
        self.output = np.dot(inputs, self.synaptic_weights) + self.biases

    def use_activation_function(self, activation_function_type):
        if activation_function_type == "softmax":
            self.output = Functions.activation_softmax(self.output)
        if activation_function_type == "ReLU":
            self.output = Functions.activation_ReLU(self.output)

    def print_output(self):
        print(self.output)
