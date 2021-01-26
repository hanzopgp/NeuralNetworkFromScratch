if __name__ == '__main__':

    # Forward pass
    x = [1.0, -2.0, 3.0] # input values
    w = [-3.0, -1.0, 2.0] # weights
    b = 1.0 # bias
    # Multiplying inputs by weights
    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]
    # Adding weighted inputs and a bias
    z = xw0 + xw1 + xw2 + b
    # ReLU activation function
    y = max(z, 0)
    print(y)

    # Backward pass
    dvalue = 1.0 # The derivative from the next layer
    # Derivative of ReLU and the chain rule
    drelu_dz = dvalue * (1. if z > 0 else 0.)
    print(drelu_dz)
    # Partial derivatives of the multiplication, the chain rule
    dsum_dxw0 = 1
    dsum_dxw1 = 1
    dsum_dxw2 = 1
    dsum_db = 1
    drelu_dxw0 = drelu_dz * dsum_dxw0
    drelu_dxw1 = drelu_dz * dsum_dxw1
    drelu_dxw2 = drelu_dz * dsum_dxw2
    drelu_db = drelu_dz * dsum_db
    print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)
    dmul_dx0 = w[0]
    dmul_dx1 = w[1]
    dmul_dx2 = w[2]
    dmul_dw0 = x[0]
    dmul_dw1 = x[1]
    dmul_dw2 = x[2]
    drelu_dx0 = drelu_dxw0 * dmul_dx0
    drelu_dw0 = drelu_dxw0 * dmul_dw0
    drelu_dx1 = drelu_dxw1 * dmul_dx1
    drelu_dw1 = drelu_dxw1 * dmul_dw1
    drelu_dx2 = drelu_dxw2 * dmul_dx2
    drelu_dw2 = drelu_dxw2 * dmul_dw2
    print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

    # Optimizing
    drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]  # Simplified value
    print(drelu_dx0)
    # Our gradients
    dx = [drelu_dx0, drelu_dx1, drelu_dx2]  # gradients on inputs
    dw = [drelu_dw0, drelu_dw1, drelu_dw2]  # gradients on weights
    db = drelu_db  # gradient on bias...just 1 bias here.
    # Checking values before backpropagation
    print(w, b)
    # Applying a fraction of the gradients to these values
    w[0] += -0.001 * dw[0]
    w[1] += -0.001 * dw[1]
    w[2] += -0.001 * dw[2]
    b += -0.001 * db
    print(w, b)

    # Another forward pass
    # Multiplying inputs by weights
    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]
    # Adding
    z = xw0 + xw1 + xw2 + b
    # ReLU activation function
    y = max(z, 0)
    print(y)

