
import numpy as np
from sklearn.datasets import make_regression

np.random.seed(0)       # same starting weights every run, so results repeat
LEARNING_RATE = 0.05    # how big a step each weight/bias update takes


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
        # Remember the original inputs; the backward pass needs them.
        self.inputs = inputs
        # Weighted sum of the inputs, plus the bias.
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, gradients):
        # Change in weights and biases.
        self.dweights = np.dot(self.inputs.T, gradients)
        # keepdims=True keeps the shape (1, n_neurons) to match self.biases.
        self.dbiases = np.sum(gradients, axis=0, keepdims=True)

        # Updated gradients to pass back to the previous layer.
        # (Computed BEFORE the weights change, using the weights from the
        # forward pass.)
        self.dinputs = np.dot(gradients, self.weights.T)

        # Update weights and biases.
        # The changes are scaled by a small learning rate. Without it the
        # steps are far too large for this dataset (targets reach ~±200)
        # and the loss explodes to inf/nan.
        self.weights -= LEARNING_RATE * self.dweights
        self.biases -= LEARNING_RATE * self.dbiases

        return self.dinputs


class Sigmoid:
    """Activation function: squashes any input into the range (0, 1)."""

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, gradients):
        # Sigmoid derivative: x(1 - x), where x is the sigmoid output
        # calculated in the forward pass.
        derivative = self.output * (1 - self.output)
        self.dinputs = gradients * derivative
        return self.dinputs


class Loss_MSE:
    """Mean-Squared-Error loss between predicted and actual values."""

    def forward(self, y_pred, y_true):
        # Reshape so predicted (n_samples, 1) and actual (n_samples,) line up
        # element-wise instead of broadcasting into an (n_samples, n_samples) mess.
        y_true = y_true.reshape(y_pred.shape)
        self.output = (y_pred - y_true) ** 2
        return self.output

    def backward(self, y_pred, y_true):
        y_true = y_true.reshape(y_pred.shape)
        n_samples = len(y_pred)

        # Loss derivative: 2(y' - y), then divide by the number of samples.
        derivative = 2 * (y_pred - y_true)
        self.dinputs = derivative / n_samples
        return self.dinputs


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------
X, y = make_regression(n_samples=100, n_features=3, n_informative=2,
                       noise=0.1, random_state=42)

# ---------------------------------------------------------------------------
# Instantiate the two hidden layers + activations, the output layer,
# and the loss object.
# ---------------------------------------------------------------------------
hidden_layer1 = Layer(n_inputs=3, n_neurons=4)   # 3 input features -> 4 neurons
activation1 = Sigmoid()

hidden_layer2 = Layer(n_inputs=4, n_neurons=4)   # 4 in (from layer 1) -> 4 neurons
activation2 = Sigmoid()

output_layer = Layer(n_inputs=4, n_neurons=1)    # 4 in (from layer 2) -> 1 output neuron

loss_function = Loss_MSE()

# ---------------------------------------------------------------------------
# Learning loop: forward pass + backward pass, repeated for 500 epochs.
# ---------------------------------------------------------------------------
epochs = 500

for epoch in range(1, epochs + 1):
    # ---- Forward pass: chain each layer's output into the next layer's input.
    hidden_layer1.forward(X)
    activation1.forward(hidden_layer1.output)

    hidden_layer2.forward(activation1.output)
    activation2.forward(hidden_layer2.output)

    output_layer.forward(activation2.output)

    # ---- Loss + average loss.
    loss_function.forward(output_layer.output, y)
    average_loss = np.mean(loss_function.output)

    # ---- Backward pass: go through the network in reverse order, passing
    #      each step's gradients to the step before it.
    loss_function.backward(output_layer.output, y)
    output_layer.backward(loss_function.dinputs)

    activation2.backward(output_layer.dinputs)
    hidden_layer2.backward(activation2.dinputs)

    activation1.backward(hidden_layer2.dinputs)
    hidden_layer1.backward(activation1.dinputs)

    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Average MSE Loss: {average_loss:.4f}")

print("Final Average MSE Loss:", average_loss)
