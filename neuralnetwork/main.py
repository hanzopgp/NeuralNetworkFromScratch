import numpy as np

from neuralnetwork.data.DataExample import DataExample
from neuralnetwork.core.NeuralNetwork import NeuralNetwork
from neuralnetwork.core.Layer import Layer
from neuralnetwork import settings

if __name__ == '__main__':

    # Example data
    data_example = DataExample(settings.NB_DATA_VALUES, settings.NB_DATA_TYPES)  # 100 points on the graph, 3 colors
    data_example.make_vertical_date()
    data_example.display_data()

    # Initializing neural network
    neural_network = NeuralNetwork(data_example.x, data_example.y, settings.LOSS_FUNCTION_TYPE)
    neural_network.add_layer(Layer(2, 3, "ReLU"))  # hidden layer + activation type
    neural_network.add_layer(Layer(3, 3, "softmax"))  # output layer + activation type

    # Optimizing neural network using randomness to lower loss
    lowest_loss = 9999999  # some initial value
    best_dense1_weights = neural_network.layers[0].synaptic_weights.copy()
    best_dense1_biases = neural_network.layers[0].biases.copy()
    best_dense2_weights = neural_network.layers[1].synaptic_weights.copy()
    best_dense2_biases = neural_network.layers[1].biases.copy()
    print("Dense 1 :")
    print(best_dense1_weights)
    print(best_dense1_biases)
    print("Dense 2:")
    print(best_dense2_weights)
    print(best_dense2_biases)
    for iteration in range(10000):
        # Generate a new set of weights for iteration
        neural_network.layers[0].synaptic_weights = 0.05 * np.random.randn(2, 3)
        neural_network.layers[0].biases = 0.05 * np.random.randn(1, 3)
        neural_network.layers[1].synaptic_weights = 0.05 * np.random.randn(3, 3)
        neural_network.layers[1].biases = 0.05 * np.random.randn(1, 3)
        neural_network.forward_layers()
        neural_network.calculate_loss()
        neural_network.calculating_accuracy()
        if neural_network.loss_value < lowest_loss:
            print('New set of weights found, iteration:', iteration,
                  'loss:', neural_network.loss_value,
                  'acc:', neural_network.accuracy)
            best_dense1_weights = neural_network.layers[0].synaptic_weights.copy()
            best_dense1_biases = neural_network.layers[0].biases.copy()
            best_dense2_weights = neural_network.layers[1].synaptic_weights.copy()
            best_dense2_biases = neural_network.layers[1].biases.copy()
            lowest_loss = neural_network.loss_value





