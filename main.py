from neuralnetwork.DataExample import DataExample
from neuralnetwork.Functions import Functions
from neuralnetwork.NeuralNetwork import NeuralNetwork
from neuralnetwork.Layer import Layer

NB_INPUTS = 2  # Inputs are x and y
NB_NEURONS = 5
ACTIVATION_FUNCTION_TYPE = "softmax"  # "softmax" / "ReLU"

if __name__ == '__main__':
    data_example = DataExample(100, 3)  # 100 points on the graph, 3 colors
    data_example.make_data()
    # data_example.display_data()

    test = [4.8, 1.21, 2.385]
    test = Functions.activation_softmax(test)
    print(test)

    neural_network = NeuralNetwork()
    neural_network.add_layer(Layer(NB_INPUTS, NB_NEURONS))
    neural_network.add_layer(Layer(NB_INPUTS, NB_NEURONS))
    neural_network.forward_layers(data_example.x)
    neural_network.use_activation_function_all(ACTIVATION_FUNCTION_TYPE)
    # neural_network.print_layers_outputs()
