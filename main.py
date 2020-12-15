from neuralnetwork.NeuralNetwork import NeuralNetwork

NB_LAYER = 1
NB_INPUTS = 4
NB_NEURONS = 5
DATA = [[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]


if __name__ == '__main__':
    neural_network = NeuralNetwork(NB_LAYER, NB_INPUTS, NB_NEURONS)
    neural_network.forward_layers(DATA)
    neural_network.print_layers_outputs()

