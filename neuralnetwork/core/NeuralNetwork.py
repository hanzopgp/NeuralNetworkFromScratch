from neuralnetwork.lossfunctions.LossCategoricalCrossentropy import LossCategoricalCrossentropy
from main import NB_LINES_PRINTED


class NeuralNetwork:

    def __init__(self, inputs, correct_outputs, loss_type):
        self.inputs = inputs
        self.correct_outputs = correct_outputs
        self.loss_type = loss_type
        self.layers = []
        self.output = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def calculate_loss(self):
        if self.loss_type == "CategoricalCrossentropy":
            loss_function = LossCategoricalCrossentropy()
            loss_function.calculate(self.layers[-1].output, self.correct_outputs)
            print(self.correct_outputs)
            print(loss_function.output)

    def forward_layers(self):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(self.inputs)
            else:
                self.layers[i].forward(self.layers[i-1].output)

    def print_infos(self):
        print("========================== TRAINING DATA ==========================")
        print(self.inputs[:NB_LINES_PRINTED])
        cpt = 1
        for i in range(len(self.layers)-1):
            print("========================== LAYER NUMBER " + str(cpt) + " ==========================")
            self.layers[i].print_output()
            cpt += 1
        print("========================== OUTPUT PROBABILITIES ==========================")
        print(self.layers[-1].print_output())
        print("========================== LOSS VALUES ==========================")
