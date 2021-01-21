from neuralnetwork.data.DataExample import DataExample
from neuralnetwork.core.NeuralNetwork import NeuralNetwork
from neuralnetwork.core.Layer import Layer

if __name__ == '__main__':
    data_example = DataExample(5, 3)  # 100 points on the graph, 3 colors
    data_example.make_data()
    # data_example.display_data()

    neural_network = NeuralNetwork(data_example.x)
    neural_network.add_layer(Layer(2, 3, "ReLU"))  # hidden layer
    neural_network.add_layer(Layer(3, 3, "softmax"))  # output layer
    neural_network.forward_layers()
    neural_network.print_layers_outputs()
