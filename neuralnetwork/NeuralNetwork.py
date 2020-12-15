import numpy as np

from neuralnetwork.Layer import Layer


class NeuralNetwork:

    def __init__(self, nb_layer, nb_inputs, nb_neurons):
        self.nb_layer = nb_layer
        self.layers = []
        for i in range(nb_layer):
            self.layers.append(Layer(nb_inputs, nb_neurons))

    def forward_layers(self, inputs):
        for i in range(self.nb_layer):
            self.layers[i].forward(inputs)

    def print_layers_outputs(self):
        for i in range(self.nb_layer):
            print("nb_layer : " + str(self.nb_layer))
            self.layers[i].print_output()