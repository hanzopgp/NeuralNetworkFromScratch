import numpy as np

from neuralnetwork.lossfunctions.LossCategoricalCrossentropy import LossCategoricalCrossentropy
from neuralnetwork import settings


class NeuralNetwork:

    def __init__(self, inputs, correct_outputs, loss_type):
        self.inputs = inputs
        self.correct_outputs = correct_outputs
        self.loss_type = loss_type
        self.loss_value = 0
        self.accuracy = 0
        self.layers = []
        self.output = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def calculate_loss(self):
        if self.loss_type == "CategoricalCrossentropy":
            loss_function = LossCategoricalCrossentropy()
            self.loss_value = loss_function.calculate(self.layers[-1].output, self.correct_outputs)

    def calculating_accuracy(self):
        predictions = np.argmax(self.layers[-1].output, axis=1)
        if len(self.correct_outputs.shape) == 2:
            self.correct_outputs = np.argmax(self.correct_outputs, axis=1)
        self.accuracy = np.mean(predictions == self.correct_outputs)

    def forward_layers(self, combined, y_true):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(self.inputs)
            elif i == len(self.layers) - 1:
                if combined:  # If combined we used this special function on the last layer
                    self.loss_value = self.layers[-1].softmax_crossentropy_forward_backward(self.layers[-2].output, y_true)
                else:
                    self.layers[-1].forward(self.layers[-2].output)
            else:
                self.layers[i].forward(self.layers[i-1].output)

    def print_infos(self):
        print("========================== TRAINING DATA ==========================")
        print(self.inputs[:settings.NB_LINES_PRINTED])
        cpt = 1
        for i in range(len(self.layers)-1):
            print("========================== LAYER NUMBER " + str(cpt) + " ==========================")
            self.layers[i].print_output()
            cpt += 1
        print("========================== OUTPUT PROBABILITIES ==========================")
        print(self.layers[-1].print_output())
        print("========================== LOSS VALUE ==========================")
        print("loss_value =", self.loss_value)
        print("accuracy =", self.accuracy)