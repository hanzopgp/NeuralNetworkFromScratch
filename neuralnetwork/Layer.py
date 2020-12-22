import numpy as np

from neuralnetwork.Functions import Functions

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
            self.output = Functions.activation_softmax(self.output)
        elif self.activation_type == "step":
            self.output = Functions.activation_step(self.output)
        elif self.activation_type == "sigmoid":
            self.output = Functions.activation_sigmoid(self.output)
        elif self.activation_type == "ReLU":
            self.output = Functions.activation_ReLU(self.output)

    def print_output(self):
        print(self.output)
