import time
import numpy as np

from neuralnetwork.data.DataExample import DataExample
from neuralnetwork.core.NeuralNetwork import NeuralNetwork
from neuralnetwork.core.Layer import Layer
from neuralnetwork import settings

if __name__ == '__main__':

    # Example data
    data_example = DataExample(settings.NB_DATA_VALUES, settings.NB_DATA_TYPES)  # 100 points on the graph, 3 colors
    if settings.DATA == "vertical":
        data_example.make_vertical_data()
    elif settings.DATA == "spiral":
        data_example.make_spiral_data()
    data_example.display_data()

    # Initializing neural network and its layers
    neural_network = NeuralNetwork(data_example.x,
                                   data_example.y,
                                   settings.LOSS_FUNCTION_TYPE,
                                   settings.OPTIMIZE_FUNCTION_TYPE)
    neural_network.add_layer(Layer(settings.NB_NN_INPUTS,
                                   settings.NB_NN_HIDDEN_LAYER_NEURONS,
                                   settings.ACTIVATION_TYPES[0]))  # hidden layer + activation type
    neural_network.add_layer(Layer(settings.NB_NN_HIDDEN_LAYER_NEURONS,
                                   settings.NB_DATA_TYPES,
                                   settings.ACTIVATION_TYPES[1]))  # output layer + activation type

    # Training loop
    start_time = time.time()
    neural_network.train()
    print("---> Training took %s seconds" % (time.time() - start_time))

    # Predicting results
    x = np.array([0, 0.5])
    result = neural_network.predict(x)
    x = np.array([0.3, 0.5])
    result = neural_network.predict(x)
    x = np.array([0.75, 0.5])
    result = neural_network.predict(x)



