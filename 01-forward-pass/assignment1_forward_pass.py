"""
 Forward pass of a neural network from scratch.

Builds a small regression network (2 hidden layers of 4 neurons each,
Sigmoid activations, 1-neuron output layer) entirely with NumPy, runs one
forward pass over a synthetic regression dataset, and reports the average
Mean-Squared-Error loss.
"""
import numpy as np
from sklearn.datasets import make_regression


class Layer:
    """
    One fully-connected layer.

    n_inputs  = how many numbers come IN to this layer
    n_neurons = how many neurons this layer HAS (how many numbers it outputs)
    """

    def __init__(self, n_inputs, n_neurons):
        # Small random starting weights (one per input->neuron connection)
        # and zero-initialized biases (one per neuron).
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Weighted sum of the inputs, plus the bias.
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


class Sigmoid:
    """Activation function: squashes any input into the range (0, 1)."""

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output


class Loss_MSE:
    """Mean-Squared-Error loss between predicted and actual values."""

    def forward(self, y_pred, y_true):
        # Reshape so predicted (n_samples, 1) and actual (n_samples,) line up
        # element-wise instead of broadcasting into an (n_samples, n_samples) mess.
        y_true = y_true.reshape(y_pred.shape)
        self.output = (y_pred - y_true) ** 2
        return self.output


# ---------------------------------------------------------------------------
# 1. Training dataset
# ---------------------------------------------------------------------------
X, y = make_regression(n_samples=100, n_features=3, n_informative=2,
                        noise=0.1, random_state=42)

# ---------------------------------------------------------------------------
# 5-7. Instantiate the two hidden layers + activations, the output layer,
#      and the loss object.
# ---------------------------------------------------------------------------
hidden_layer1 = Layer(n_inputs=3, n_neurons=4)   # 3 input features -> 4 neurons
activation1 = Sigmoid()

hidden_layer2 = Layer(n_inputs=4, n_neurons=4)   # 4 in (from layer 1) -> 4 neurons
activation2 = Sigmoid()

output_layer = Layer(n_inputs=4, n_neurons=1)    # 4 in (from layer 2) -> 1 output neuron

loss_function = Loss_MSE()

# ---------------------------------------------------------------------------
# 8. Forward pass: chain each layer's output into the next layer's input.
# ---------------------------------------------------------------------------
hidden_layer1.forward(X)
activation1.forward(hidden_layer1.output)

hidden_layer2.forward(activation1.output)
activation2.forward(hidden_layer2.output)

output_layer.forward(activation2.output)

# ---------------------------------------------------------------------------
# 9-10. Loss + average loss.
# ---------------------------------------------------------------------------
loss_function.forward(output_layer.output, y)
average_loss = np.mean(loss_function.output)

print("Average MSE Loss:", average_loss)
