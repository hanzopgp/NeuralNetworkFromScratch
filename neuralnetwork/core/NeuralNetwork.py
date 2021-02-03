import numpy as np

from neuralnetwork.activationfunctions.ActivationReLU import ActivationReLU
from neuralnetwork.activationxlossfunctions.ActivationSoftmaxLossCategoricalCrossentropy import ActivationSoftmaxLossCategoricalCrossentropy
from neuralnetwork.lossfunctions.LossCategoricalCrossentropy import LossCategoricalCrossentropy
from neuralnetwork import settings
from neuralnetwork.optimizerfunctions.OptimizerSGD import OptimizerSGD


class NeuralNetwork:

    def __init__(self, inputs, correct_outputs, loss_type, optimize_type):
        self.inputs = inputs
        self.correct_outputs = correct_outputs
        self.loss_type = loss_type
        self.optimize_type = optimize_type
        self.loss_value = 0
        self.accuracy = 0
        self.layers = []
        self.output = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_layers(self, combined):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(self.inputs)
            elif i == len(self.layers) - 1:
                if combined:  # If combined we use this special function on the last layer
                    self.loss_value = self.layers[-1].softmax_crossentropy_forward_backward(self.layers[-2].output, self.correct_outputs)
                else:
                    self.layers[-1].forward(self.layers[-2].output)
            else:
                self.layers[i].forward(self.layers[i-1].output)

    def backward_layers(self, previous_output):  # Make it scalable
        softmax_crossentropy = ActivationSoftmaxLossCategoricalCrossentropy()
        relu = ActivationReLU()
        softmax_crossentropy.backward(previous_output, self.correct_outputs)
        self.layers[-1].backward(softmax_crossentropy.dinputs)
        relu.backward(self.layers[-1].dinputs)
        self.layers[-2].backward(relu.dinputs)

    def calculate_loss(self):
        if self.loss_type == "CategoricalCrossentropy":
            loss_function = LossCategoricalCrossentropy()
            self.loss_value = loss_function.calculate(self.layers[-1].output, self.correct_outputs)

    def calculating_accuracy(self):
        predictions = np.argmax(self.layers[-1].output, axis=1)
        if len(self.correct_outputs.shape) == 2:
            self.correct_outputs = np.argmax(self.correct_outputs, axis=1)
        self.accuracy = np.mean(predictions == self.correct_outputs)

    def optimize_layers(self, learning_rate):
        for i in range(len(self.layers)):
            if self.optimize_type == "SGD":
                optimizer = OptimizerSGD(learning_rate)
                optimizer.update_params(self.layers[i])

    def train(self):
        # Training loop
        for epoch in range(settings.NB_NN_EPOCHS):
            # Forward pass
            if settings.COMBINED_SOFTMAX_CROSSENTROPY:
                self.forward_layers(combined=True)
            else:
                self.forward_layers(combined=False)
                self.calculate_loss()
            self.calculating_accuracy()
            # Printing epoch network values
            if not epoch % settings.NB_NN_EPOCHS_STEP:
                # print(neural_network.print_infos())
                print(f'epoch: {epoch}, ' +
                      f'acc: {self.accuracy:.3f}, ' +
                      f'loss: {self.loss_value:.3f}')
            # Backpropagation
            prediction = self.layers[-1].output.copy()
            self.backward_layers(prediction)  # Giving the outputs of the neural network and correct outputs
            # Updating weights and biases
            self.optimize_layers(settings.LEARNING_RATE)

    def predict(self, x):
        self.inputs = x
        print("input :", x)
        self.forward_layers(settings.COMBINED_SOFTMAX_CROSSENTROPY)
        print("probabilities :", self.layers[-1].output)
        color = ["blue", "red", "green"]
        color_index = int(np.argmax(self.layers[-1].output))
        print("color predicted :", color[color_index])
        print("confidence :", int(np.max(self.layers[-1].output)*100), "%")

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
        print("========================== CORRECT OUTPUTS ==========================")
        print(self.correct_outputs)
        print("========================== LOSS VALUE ==========================")
        print("loss_value =", self.loss_value)
        print("accuracy =", self.accuracy)