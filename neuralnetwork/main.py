import time

from neuralnetwork.data.DataExample import DataExample
from neuralnetwork.core.NeuralNetwork import NeuralNetwork
from neuralnetwork.core.Layer import Layer
from neuralnetwork import settings


if __name__ == '__main__':

    # Example data
    data_example = DataExample(settings.NB_DATA_VALUES, settings.NB_DATA_TYPES)  # 100 points on the graph, 3 colors
    data_example.make_vertical_date()
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

    # Keeping track of training duration
    start_time = time.time()

    # Training loop
    for epoch in range(10_001):

        # Forward pass
        if settings.COMBINED_SOFTMAX_CROSSENTROPY:
            neural_network.forward_layers(combined=True)
        else:
            neural_network.forward_layers(combined=False)
            neural_network.calculate_loss()
        neural_network.calculating_accuracy()

        # Printing epoch network values
        if not epoch % 100:
            print(neural_network.print_infos())
            print(f'epoch: {epoch}, ' +
                  f'acc: {neural_network.accuracy:.3f}, ' +
                  f'loss: {neural_network.loss_value:.3f}')

        # Backpropagation
        prediction = neural_network.layers[-1].output.copy()
        neural_network.backward_layers(prediction)  # Giving the outputs of the neural network and correct outputs
        neural_network.optimize_layers(settings.LEARNING_RATE)

    print("---> Training took %s seconds" % (time.time() - start_time))



