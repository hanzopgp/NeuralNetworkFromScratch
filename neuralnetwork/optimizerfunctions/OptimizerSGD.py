class OptimizerSGD:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.synaptic_weights += - self.learning_rate * layer.dweights
        layer.biases += - self.learning_rate * layer.dbiases
