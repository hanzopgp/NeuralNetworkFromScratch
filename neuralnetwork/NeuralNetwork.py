class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.output = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_layers(self, inputs):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(inputs)
            else:
                self.layers[i].forward(self.layers[i-1].output)

    def print_layers_outputs(self):
        cpt = 1
        for i in range(len(self.layers)):
            print("nb_layer : " + str(cpt))
            self.layers[i].print_output()
            cpt += 1
