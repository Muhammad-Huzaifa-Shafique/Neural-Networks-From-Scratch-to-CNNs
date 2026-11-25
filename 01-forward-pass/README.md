# 1. Forward pass from scratch

A small regression network written with NumPy only: a dense layer, a sigmoid activation and a mean-squared-error loss, chained into a 3-4-4-1 network (three inputs, two hidden layers of four sigmoid neurons, one output neuron).

```
python assignment1_forward_pass.py
```

## What it does

- Generates 100 samples with `make_regression` (3 features, 2 informative, noise 0.1, `random_state=42`).
- Passes them through the network once and prints the average mean-squared error.

## Code map

| Class | Job |
|---|---|
| `Layer` | Holds the weights (0.1 times a standard normal) and zero biases, and computes `inputs @ weights + biases`. |
| `Sigmoid` | Squashes each value into (0, 1). |
| `Loss_MSE` | Squared difference between prediction and target, reshaped so the shapes line up instead of broadcasting into an n by n matrix. |

## What to expect

The weights are random and not seeded, so the printed loss changes a little from run to run. Three runs printed 795.94, 796.14 and 796.25. The 100 targets have a variance of about 795.5, so an untrained network that outputs values near zero is about as far off as one that always predicts the mean. Nothing is trained in this assignment: the point is the shape of the computation, which assignment 2 extends with a backward pass.
