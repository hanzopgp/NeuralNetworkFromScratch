import numpy as np


if __name__ == '__main__':

    softmax_output = [0.7, 0.1, 0.2]

    print("before reshape")
    print(softmax_output)
    softmax_output = np.array(softmax_output).reshape(-1, 1)
    print("after reshape")
    print(softmax_output)

    print("softmax_ouput shape eye")
    print(np.eye(softmax_output.shape[0]))
    print("softmax_output multiplied by the eye")
    print(softmax_output * np.eye(softmax_output.shape[0]))

    print("can optimize using np.diagflat")
    print(np.diagflat(softmax_output))

    print("dot product of softmax_output and softmax_output.T")
    print(np.dot(softmax_output, softmax_output.T))

    print("subtracting the diagflat by the dot product")
    print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))

    print("this is our jacobian matrix")


