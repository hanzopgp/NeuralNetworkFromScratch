from neuralnetwork.DataExample import DataExample
from neuralnetwork.NeuralNetwork import NeuralNetwork
from neuralnetwork.Layer import Layer

if __name__ == '__main__':
    data_example = DataExample(5, 3)  # 100 points on the graph, 3 colors
    data_example.make_data()
    # data_example.display_data()

    neural_network = NeuralNetwork(data_example.x)
    neural_network.add_layer(Layer(2, 3, "ReLU"))
    neural_network.add_layer(Layer(3, 3, "softmax"))
    neural_network.forward_layers()
    neural_network.print_layers_outputs()
