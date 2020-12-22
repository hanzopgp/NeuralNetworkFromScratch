class NeuralNetwork:

    def __init__(self, inputs):
        self.inputs = inputs
        self.layers = []
        self.output = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_layers(self):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(self.inputs)
            else:
                self.layers[i].forward(self.layers[i-1].output)

    def print_layers_outputs(self):
        print("========================== TRAINING DATA ==========================")
        print(self.inputs)
        cpt = 1
        for i in range(len(self.layers) - 1):
            print("========================== LAYER NUMBER " + str(cpt) + " ==========================")
            self.layers[i].print_output()
            cpt += 1
        print("========================== OUTPUT PROBABILITIES ==========================")
        print(self.layers[-1].print_output())