import numpy as np
np.random.seed(0)


class Layer:

    def __init__(self, nb_inputs, nb_neurons):
        self.synaptic_weights = np.random.randn(nb_inputs, nb_neurons)/10 #reversed nb_neurons and nb_inputs to avoid using transpose
        self.biases = np.zeros((1, nb_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.synaptic_weights) + self.biases

    def print_output(self):
        print(self.output)
