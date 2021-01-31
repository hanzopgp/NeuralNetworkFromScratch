import numpy as np
from timeit import timeit
import nnfs

from neuralnetwork.activationfunctions.ActivationSoftmax import ActivationSoftmax
from neuralnetwork.activationxlossfunctions.ActivationSoftmaxLossCategoricalCrossentropy import ActivationSoftmaxLossCategoricalCrossentropy
from neuralnetwork.lossfunctions.LossCategoricalCrossentropy import LossCategoricalCrossentropy

nnfs.init()


def f1():
    # Combined
    sofmax_crossentropy = ActivationSoftmaxLossCategoricalCrossentropy()
    sofmax_crossentropy.backward(softmax_outputs, class_targets)
    dvaluesCombined = sofmax_crossentropy.dinputs


def f2():
    # Not combined
    activation = ActivationSoftmax()
    loss = LossCategoricalCrossentropy()
    activation.output = softmax_outputs
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinputs)
    dvaluesNotCombined = activation.dinputs


if __name__ == '__main__':

    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])
    class_targets = np.array([0, 1, 1])

    t1 = timeit(lambda: f1(), number=10000)
    t2 = timeit(lambda: f2(), number=10000)
    print("Saving", t2/t1, "s")

