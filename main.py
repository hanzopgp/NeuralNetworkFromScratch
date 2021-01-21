from neuralnetwork.data.DataExample import DataExample
from neuralnetwork.core.NeuralNetwork import NeuralNetwork
from neuralnetwork.core.Layer import Layer
import settings

if __name__ == '__main__':
    data_example = DataExample(settings.NB_DATA_VALUES, settings.NB_DATA_TYPES)  # 100 points on the graph, 3 colors
    data_example.make_data()
    # data_example.display_data()

    neural_network = NeuralNetwork(data_example.x, data_example.y, settings.LOSS_FUNCTION_TYPE)  # giving data and loss function type to nn
    neural_network.add_layer(Layer(2, 3, "ReLU"))  # hidden layer + activation type
    neural_network.add_layer(Layer(3, 3, "softmax"))  # output layer + activation type
    neural_network.forward_layers()
    neural_network.print_infos()
    neural_network.calculate_loss()
