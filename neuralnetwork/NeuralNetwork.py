class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_layers(self, inputs):
        for i in range(len(self.layers)):
            self.layers[i].forward(inputs)

    def use_activation_function_all(self, activation_function_type):
        for i in range(len(self.layers)):
            self.layers[i].use_activation_function(activation_function_type)

    def print_layers_outputs(self):
        cpt = 0
        for i in range(len(self.layers)):
            print("nb_layer : " + str(cpt+1))
            self.layers[i].print_output()
            cpt += 1