import numpy as np

from neuralnetwork.Functions import Functions

np.random.seed(0)


class Layer:

    def __init__(self, nb_inputs, nb_neurons):
        self.synaptic_weights = np.random.randn(nb_inputs, nb_neurons)/10 # reversed nb_neurons and nb_inputs to avoid using transpose
        self.biases = np.zeros((1, nb_neurons))
        self.output = 0

    def forward(self, inputs):
        self.output = np.dot(inputs, self.synaptic_weights) + self.biases

    def use_activation_function(self):
        self.output = Functions.Activation_ReLU(self.output)

    def print_output(self):
        print(self.output)
