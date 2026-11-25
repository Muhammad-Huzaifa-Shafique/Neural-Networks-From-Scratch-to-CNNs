"""
Best-performing model from the Assignment 4 CNN experiments: 02_deeper_network
Test accuracy: 0.7454

Adds a third conv layer (128 filters) to test whether more depth lets the network build up more complex, hierarchical features (edges -> textures -> object parts) than 2 layers can.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

model = keras.Sequential(name="02_deeper_network")
model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding="same"))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), padding="same"))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation("relu"))
model.add(layers.Dense(10, activation="softmax"))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=2)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
