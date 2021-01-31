import numpy as np
import nnfs

from neuralnetwork.activationfunctions.ActivationSoftmax import ActivationSoftmax
from neuralnetwork.activationxlossfunctions.ActivationSoftmaxLossCategoricalCrossentropy import ActivationSoftmaxLossCategoricalCrossentropy
from neuralnetwork.lossfunctions.LossCategoricalCrossentropy import LossCategoricalCrossentropy

nnfs.init()

if __name__ == '__main__':

    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])
    class_targets = np.array([0, 1, 1])

    # Combined
    sofmax_crossentropy = ActivationSoftmaxLossCategoricalCrossentropy()
    sofmax_crossentropy.backward(softmax_outputs, class_targets)
    dvaluesCombined = sofmax_crossentropy.dinputs
    print("dvaluesCombined :")
    print(dvaluesCombined)

    # Not combined
    activation = ActivationSoftmax()
    loss = LossCategoricalCrossentropy()
    activation.output = softmax_outputs
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinputs)
    dvaluesNotCombined = activation.dinputs
    print("dvaluesNotCombined :")
    print(dvaluesNotCombined)
