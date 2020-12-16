from neuralnetwork.DataExample import DataExample
from neuralnetwork.Layer import Layer
from neuralnetwork.NeuralNetwork import NeuralNetwork

NB_INPUTS = 2  # Inputs are x and y
NB_NEURONS = 5

if __name__ == '__main__':
    data_example = DataExample(100, 3) # 100 points on the graph, 3 colors
    data_example.make_data()
    #d ata_example.display_data()

    neural_network = NeuralNetwork()
    neural_network.add_layer(Layer(NB_INPUTS, NB_NEURONS))
    # neural_network.add_layer(Layer(NB_INPUTS, NB_NEURONS))
    neural_network.forward_layers(data_example.x)
    neural_network.use_activation_function_all()
    neural_network.print_layers_outputs()
