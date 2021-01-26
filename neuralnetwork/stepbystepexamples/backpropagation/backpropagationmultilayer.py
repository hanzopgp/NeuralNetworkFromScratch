import numpy as np


if __name__ == '__main__':
    # Passed in gradient from the next layer
    # for the purpose of this example we're going to use
    # an array of an incremental gradient values
    dvalues = np.array([[1., 1., 1.],
                        [2., 2., 2.],
                        [3., 3., 3.]])

    # WEIGHTS
    # We have 3 sets of weights - one set for each neuron
    # We have 4 inputs, thus 4 weights
    # recall that we keep weights transposed
    weights = np.array([[0.2, 0.8, -0.5, 1],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]]).T
    # Sum weights of given input
    # and multiply by the passed in gradient for this neuron
    dinputs = np.dot(dvalues, weights.T)
    print("dinputs :")
    print(dinputs)

    # INPUTS
    # We have 3 sets of inputs - samples
    inputs = np.array([[1, 2, 3, 2.5],
                       [2., 5., -1., 2],
                       [-1.5, 2.7, 3.3, -0.8]])
    # sum weights of given input
    # and multiply by the passed in gradient for this neuron
    dweights = np.dot(inputs.T, dvalues)
    print("dweights :")
    print(dweights)

    # BIASES
    # One bias for each neuron
    # biases are the row vector with a shape (1, neurons)
    biases = np.array([[2, 3, 0.5]])
    # dbiases - sum values, do this over samples (first axis), keepdims
    # since this by default will produce a plain list -
    # we explained this in the chapter 4
    dbiases = np.sum(dvalues, axis=0, keepdims=True)  # keepdims lets us keep the gradient as a row vector
    print("dbiases :")
    print(dbiases)

    # RELU
    # Example layer output
    z = np.array([[1, 2, -3, -4],
                  [2, -7, -1, 3],
                  [-1, 2, 5, -1]])
    dvalues = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]])
    # ReLU activation's derivative
    drelu = np.zeros_like(z)
    drelu[z > 0] = 1
    print("drelu :")
    print(drelu)
    # The chain rule
    drelu *= dvalues
    print("drelu + chain rule :")
    print(drelu)

    # RELU Optimized
    drelu = dvalues.copy()
    drelu[z <= 0] = 0
    print("optimized drelu + chain rule :")
    print(drelu)




