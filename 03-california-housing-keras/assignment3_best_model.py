"""
Assignment 3: Best model (Model 5: More epochs with early stopping)

Dense neural network that predicts California housing prices.
Test MAPE when the experiments were run: 18.91%
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers

SEED = 42
keras.utils.set_random_seed(SEED)

CONFIG = {'hidden_layers': [128, 64, 32],
 'activation': 'relu',
 'learning_rate': 0.001,
 'loss': 'mse',
 'batch_size': 32,
 'epochs': 200,
 'dropout': 0.0,
 'batch_norm': False,
 'target': 'scaled',
 'early_stopping': True,
 'reduce_lr': False}

# 1. Load the California Housing dataset
(X_train_full, y_train_full), (X_test, y_test) = datasets.california_housing.load_data(
    version="large", path="california_housing.npz", test_split=0.2, seed=113)
X_train_full, X_test = X_train_full.astype("float32"), X_test.astype("float32")
y_train_full, y_test = y_train_full.astype("float32"), y_test.astype("float32")

# Validation split (15% of the training data)
order = np.random.default_rng(SEED).permutation(len(X_train_full))
n_val = int(len(order) * 0.15)
X_val, y_val = X_train_full[order[:n_val]], y_train_full[order[:n_val]]
X_tr, y_tr = X_train_full[order[n_val:]], y_train_full[order[n_val:]]

# 2. Min-max scaling (fit on training data only)
x_min, x_max = X_tr.min(axis=0), X_tr.max(axis=0)
scale = lambda X: (X - x_min) / (x_max - x_min)
X_tr, X_val, X_test = scale(X_tr), scale(X_val), scale(X_test)

# Target transform
if CONFIG["target"] == "log":
    encode, decode = np.log, np.exp
else:
    encode, decode = (lambda y: y / 100000.0), (lambda y: y * 100000.0)

# 3. Define the dense network
model = keras.Sequential(name="housing_dnn")
model.add(keras.Input(shape=(8,)))
for units in CONFIG["hidden_layers"]:
    model.add(layers.Dense(units))
    if CONFIG["batch_norm"]:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation(CONFIG["activation"]))
    if CONFIG["dropout"] > 0:
        model.add(layers.Dropout(CONFIG["dropout"]))
model.add(layers.Dense(1, activation="linear"))

# 4. Compile with the MAPE metric
model.compile(optimizer=keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
              loss=CONFIG["loss"],
              metrics=["mean_absolute_percentage_error"])
model.summary()

callbacks = []
if CONFIG["early_stopping"]:
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True))
if CONFIG["reduce_lr"]:
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6))

# 5. Train the model on the training data
model.fit(X_tr, encode(y_tr), validation_data=(X_val, encode(y_val)),
          epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"],
          callbacks=callbacks, verbose=2)

# 6. Evaluate on the test data (MAPE measured in dollars)
pred = decode(model.predict(X_test, verbose=0).ravel())
test_mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
print(f"Test MAPE: {test_mape:.2f}%")
print(f"Test MAE:  ${np.mean(np.abs(y_test - pred)):,.0f}")
