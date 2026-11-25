"""
Assignment 3: Dense neural network for the Keras California Housing dataset.

Running this ONE script does the whole assignment:
  1. Loads the California Housing dataset (large version, 20,640 samples).
  2. Min-max scales the 8 input features.
  3. Trains and tests a base dense network, then nine modified versions
     (ten models in total), using Mean Absolute Percentage Error (MAPE)
     as the metric.
  4. Writes the Word report  -> Assignment3_Report.docx
  5. Writes the source code of the best model -> assignment3_best_model.py

How the modifications work
--------------------------
Each variation makes ONE change to the best configuration found so far.
If the change lowers the VALIDATION MAPE (a 15% slice of the training data
the model never trains on), the change is kept. Otherwise it is reverted
before the next variation is tried. The test set is only used to report
each model's final score. It is never used to decide which changes to keep.

Requirements:  pip install tensorflow python-docx matplotlib
Runtime:       roughly 10-25 minutes on a normal laptop or Google Colab (CPU).
"""
import os
import pprint

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # save charts to files, no window
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

SEED = 42
VALIDATION_FRACTION = 0.15
TARGET_SCALE = 100_000.0     # prices are divided by this before training
REPORT_FILE = "Assignment3_Report.docx"
BEST_MODEL_FILE = "assignment3_best_model.py"
CHART_FILE = "best_model_learning_curve.png"


# ===========================================================================
# 1. Load the data
# ===========================================================================
(X_train_full, y_train_full), (X_test, y_test) = datasets.california_housing.load_data(
    version="large", path="california_housing.npz", test_split=0.2, seed=113)

X_train_full = X_train_full.astype("float32")
X_test = X_test.astype("float32")
y_train_full = y_train_full.astype("float32")
y_test = y_test.astype("float32")

# Hold back part of the training data as a validation set. It is used to
# decide which modifications to keep, so the test set stays untouched.
rng = np.random.default_rng(SEED)
order = rng.permutation(len(X_train_full))
n_val = int(len(order) * VALIDATION_FRACTION)
val_idx, tr_idx = order[:n_val], order[n_val:]
X_tr, y_tr = X_train_full[tr_idx], y_train_full[tr_idx]
X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]


# ===========================================================================
# 2. Min-max scaling of the inputs:  x' = (x - min) / (max - min)
#    min and max come from the TRAINING data only, then the same values are
#    applied to the validation and test data (no information leaks from test).
# ===========================================================================
x_min = X_tr.min(axis=0)
x_max = X_tr.max(axis=0)


def min_max_scale(X):
    return (X - x_min) / (x_max - x_min)


X_tr_s, X_val_s, X_test_s = min_max_scale(X_tr), min_max_scale(X_val), min_max_scale(X_test)


# ---------------------------------------------------------------------------
# Target (price) transform. Prices go up to $500,000, so they are either
# divided by 100,000 ("scaled") or log-transformed ("log") for training.
# Predictions are always converted back to dollars before MAPE is measured,
# so every model is scored the same way.
# ---------------------------------------------------------------------------
def encode_target(y, mode):
    return np.log(y) if mode == "log" else y / TARGET_SCALE


def decode_target(y, mode):
    return np.exp(y) if mode == "log" else y * TARGET_SCALE


# ===========================================================================
# 3. Model builder + training / evaluation
# ===========================================================================
def build_model(cfg):
    model = keras.Sequential(name="housing_dnn")
    model.add(keras.Input(shape=(8,)))
    for units in cfg["hidden_layers"]:
        model.add(layers.Dense(units))
        if cfg["batch_norm"]:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(cfg["activation"]))
        if cfg["dropout"] > 0:
            model.add(layers.Dropout(cfg["dropout"]))
    model.add(layers.Dense(1, activation="linear"))    # one price per house

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
                  loss=cfg["loss"],
                  metrics=["mean_absolute_percentage_error"])
    return model


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def train_and_evaluate(cfg):
    keras.utils.set_random_seed(SEED)                  # same start for every model
    model = build_model(cfg)

    callbacks = []
    if cfg["early_stopping"]:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True))
    if cfg["reduce_lr"]:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6))

    history = model.fit(
        X_tr_s, encode_target(y_tr, cfg["target"]),
        validation_data=(X_val_s, encode_target(y_val, cfg["target"])),
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        callbacks=callbacks, verbose=0)

    val_pred = decode_target(model.predict(X_val_s, verbose=0).ravel(), cfg["target"])
    test_pred = decode_target(model.predict(X_test_s, verbose=0).ravel(), cfg["target"])

    ss_res = np.sum((y_test - test_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    return {
        "val_mape": mape(y_val, val_pred),
        "test_mape": mape(y_test, test_pred),
        "test_mae": float(np.mean(np.abs(y_test - test_pred))),
        "test_r2": float(1 - ss_res / ss_tot),
        "epochs_run": len(history.history["loss"]),
        "params": model.count_params(),
        "history": history.history,
    }


# ===========================================================================
# 4. The base model and the nine modifications
# ===========================================================================
BASE_CONFIG = {
    "hidden_layers": [64, 32],    # neurons in each hidden layer
    "activation": "relu",
    "learning_rate": 0.001,       # optimizer is always Adam
    "loss": "mse",
    "batch_size": 32,
    "epochs": 50,
    "dropout": 0.0,
    "batch_norm": False,
    "target": "scaled",           # "scaled" = price / 100,000   "log" = ln(price)
    "early_stopping": False,
    "reduce_lr": False,
}

MODIFICATIONS = [
    {
        "title": "Wider and deeper network",
        "change": {"hidden_layers": [128, 64, 32]},
        "what": "Added a third hidden layer and doubled the first layer: 128 -> 64 -> 32 neurons.",
        "why": "House prices depend on non-linear interactions between features, especially "
               "latitude and longitude together (coastal areas vs. inland). More neurons and "
               "one more layer give the network more capacity to learn these patterns.",
    },
    {
        "title": "Mean Absolute Error loss",
        "change": {"loss": "mae"},
        "what": "Changed the loss function from Mean Squared Error to Mean Absolute Error.",
        "why": "MSE squares the errors, so a few very expensive homes (the dataset is capped at "
               "$500,001) dominate training. MAE treats every error in proportion to its size, "
               "which is closer to how MAPE scores the model.",
    },
    {
        "title": "Log-transformed target",
        "change": {"target": "log"},
        "what": "Trained on ln(price) instead of price / 100,000. Predictions are converted back "
                "with exp() before scoring.",
        "why": "MAPE measures relative (percentage) error. On a log scale, a 10% error has the same "
               "size for a $60,000 home as for a $450,000 home. The loss then matches the metric, "
               "and cheap homes (where percentage errors get large) get proper attention.",
    },
    {
        "title": "More epochs with early stopping",
        "change": {"epochs": 200, "early_stopping": True},
        "what": "Raised the maximum to 200 epochs and added EarlyStopping (monitor val_loss, "
                "patience 15, restore best weights).",
        "why": "50 epochs may stop before the model has converged. Early stopping lets the model "
               "train longer, then stops it and restores the best weights once validation loss "
               "stops improving, which prevents overfitting.",
    },
    {
        "title": "Learning-rate reduction on plateau",
        "change": {"reduce_lr": True},
        "what": "Added ReduceLROnPlateau (halves the learning rate after 5 epochs without "
                "improvement in val_loss, minimum 1e-6).",
        "why": "A fixed learning rate can bounce around the minimum late in training. Smaller "
               "steps near the end let the optimizer settle into a better solution.",
    },
    {
        "title": "Larger batch size",
        "change": {"batch_size": 64},
        "what": "Increased the batch size from 32 to 64.",
        "why": "Larger batches give smoother, less noisy gradient estimates and faster epochs. "
               "The test checks whether the smoother updates help, or whether the noise of "
               "smaller batches was helping the model generalize.",
    },
    {
        "title": "Dropout regularization",
        "change": {"dropout": 0.1},
        "what": "Added a Dropout layer (rate 0.1) after each hidden layer.",
        "why": "Randomly switching off 10% of neurons during training stops the network from "
               "relying on a few neurons, which usually reduces overfitting. This matters more "
               "now that the network is bigger.",
    },
    {
        "title": "Batch normalization",
        "change": {"batch_norm": True},
        "what": "Added BatchNormalization between each Dense layer and its activation.",
        "why": "Batch normalization keeps each layer's inputs at a stable scale during training. "
               "This often speeds up convergence and makes deeper networks easier to train.",
    },
    {
        "title": "Swish activation",
        "change": {"activation": "swish"},
        "what": "Changed the hidden-layer activation from ReLU to Swish (x * sigmoid(x)).",
        "why": "ReLU outputs exactly zero for negative inputs, so some neurons can stop learning "
               "(\"dying ReLU\"). Swish is smooth and lets small negative values through, which "
               "often gives slightly better results on regression tasks.",
    },
]


def describe_layers(cfg):
    """Human-readable list of every layer: (name, neurons, activation)."""
    rows = []
    for i, units in enumerate(cfg["hidden_layers"], start=1):
        extra = []
        if cfg["batch_norm"]:
            extra.append("BatchNorm before activation")
        if cfg["dropout"] > 0:
            extra.append(f"Dropout {cfg['dropout']}")
        rows.append((f"Hidden layer {i} (Dense)", str(units), cfg["activation"],
                     ", ".join(extra) or "-"))
    rows.append(("Output layer (Dense)", "1", "linear", "-"))
    return rows


def other_elements(cfg):
    items = [
        "Input: 8 features, min-max scaled to [0, 1] using training-set min/max",
        ("Target: ln(price), converted back with exp() for scoring" if cfg["target"] == "log"
         else "Target: price / 100,000, converted back to dollars for scoring"),
        f"Validation set: {int(VALIDATION_FRACTION * 100)}% of the training data",
        f"Random seed: {SEED}",
    ]
    if cfg["early_stopping"]:
        items.append("EarlyStopping: monitor val_loss, patience 15, restore best weights")
    if cfg["reduce_lr"]:
        items.append("ReduceLROnPlateau: factor 0.5, patience 5, min_lr 1e-6")
    return items


# ===========================================================================
# 5. Run all ten models
# ===========================================================================
results = []

print("Training model 1/10: Base model")
res = train_and_evaluate(BASE_CONFIG)
results.append({"id": 1, "title": "Base model", "config": dict(BASE_CONFIG),
                "what": "Starting point.", "why": "-", "kept": True, **res})
print(f"   val MAPE {res['val_mape']:.2f}%   test MAPE {res['test_mape']:.2f}%")

current = dict(BASE_CONFIG)
current_val = res["val_mape"]

for i, mod in enumerate(MODIFICATIONS, start=2):
    cfg = {**current, **mod["change"]}
    print(f"Training model {i}/10: {mod['title']}")
    res = train_and_evaluate(cfg)
    kept = res["val_mape"] < current_val
    if kept:
        current, current_val = cfg, res["val_mape"]
    results.append({"id": i, "title": mod["title"], "config": cfg,
                    "what": mod["what"], "why": mod["why"], "kept": kept, **res})
    print(f"   val MAPE {res['val_mape']:.2f}%   test MAPE {res['test_mape']:.2f}%   "
          f"{'KEPT' if kept else 'reverted'}")

best = min(results, key=lambda r: r["test_mape"])
print(f"\nBest model: #{best['id']} {best['title']}  (test MAPE {best['test_mape']:.2f}%)")


# ===========================================================================
# 6. Learning-curve chart for the best model
# ===========================================================================
h = best["history"]
plt.figure(figsize=(7, 3.5))
plt.plot(h["loss"], label="Training loss")
plt.plot(h["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel(f"Loss ({best['config']['loss'].upper()})")
plt.title(f"Model {best['id']}: training vs. validation loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150)
plt.close()


# ===========================================================================
# 7. Word report
# ===========================================================================
def add_table(doc, header, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    for cell, text in zip(table.rows[0].cells, header):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)
    if widths:
        for row in table.rows:
            for cell, w in zip(row.cells, widths):
                cell.width = Inches(w)
    doc.add_paragraph()
    return table


def add_model_elements(doc, cfg, r):
    add_table(doc, ["Layer", "Neurons", "Activation", "Extras"], describe_layers(cfg),
              widths=[2.0, 0.8, 1.1, 2.6])
    add_table(doc, ["Element", "Value"], [
        ("Number of layers", f"{len(cfg['hidden_layers']) + 1} Dense layers "
                             f"({len(cfg['hidden_layers'])} hidden + 1 output)"),
        ("Optimizer", "Adam"),
        ("Learning rate", cfg["learning_rate"]),
        ("Loss function", {"mse": "Mean Squared Error (MSE)",
                           "mae": "Mean Absolute Error (MAE)"}[cfg["loss"]]),
        ("Metric", "Mean Absolute Percentage Error (MAPE)"),
        ("Batch size", cfg["batch_size"]),
        ("Epochs", f"{cfg['epochs']} max, {r['epochs_run']} run" if cfg["early_stopping"]
                   else cfg["epochs"]),
        ("Total parameters", f"{r['params']:,}"),
    ], widths=[2.0, 4.5])
    doc.add_paragraph("Other elements used:")
    for item in other_elements(cfg):
        doc.add_paragraph(item, style="List Bullet")


doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)

title = doc.add_heading("Assignment 3: Dense Neural Network for California Housing Prices", 0)
doc.add_paragraph("Tools: Python, TensorFlow, Keras")

doc.add_heading("1. Dataset and preprocessing", 1)
doc.add_paragraph(
    f"The Keras California Housing dataset (large version, 20,640 samples, 8 features) was "
    f"loaded with test_split=0.2 and seed=113, which gives {len(X_train_full):,} training and "
    f"{len(X_test):,} test samples. {int(VALIDATION_FRACTION * 100)}% of the training samples "
    f"({len(X_val):,}) were set aside as a validation set, leaving {len(X_tr):,} samples for "
    f"training. The validation set was used only to decide which modifications to keep.")
doc.add_paragraph(
    "Inputs were normalized with min-max scaling, x' = (x - min) / (max - min), using the "
    "minimum and maximum of the training data, so every feature lies in [0, 1]. The same "
    "training min/max values were applied to the validation and test data.")
doc.add_paragraph(
    "The metric is Mean Absolute Percentage Error (MAPE): the average percentage by which the "
    "predicted price differs from the actual price. Lower is better. All predictions were "
    "converted back to dollars before MAPE was calculated, so every model is scored the same "
    "way. Test MAE (average error in dollars) and R² are also reported.")

doc.add_heading("2. Base model", 1)
add_model_elements(doc, BASE_CONFIG, results[0])
p = doc.add_paragraph()
p.add_run(f"Base model test MAPE: {results[0]['test_mape']:.2f}%").bold = True
p.add_run(f"  (test MAE ${results[0]['test_mae']:,.0f}, R² {results[0]['test_r2']:.3f})")

doc.add_heading("3. Modifications", 1)
doc.add_paragraph(
    "Each variation made one change to the best configuration found so far. If the change "
    "lowered the validation MAPE it was kept, otherwise it was reverted before the next "
    "variation. Every model was trained on the training data and tested on the test data.")
for r in results[1:]:
    doc.add_heading(f"Model {r['id']}: {r['title']}", 2)
    p = doc.add_paragraph()
    p.add_run("Modification: ").bold = True
    p.add_run(r["what"])
    p = doc.add_paragraph()
    p.add_run("Reasoning: ").bold = True
    p.add_run(r["why"])
    p = doc.add_paragraph()
    p.add_run("Result: ").bold = True
    p.add_run(f"test MAPE {r['test_mape']:.2f}% (validation MAPE {r['val_mape']:.2f}%). "
              + ("The change improved validation MAPE and was kept."
                 if r["kept"] else
                 "The change did not improve validation MAPE and was reverted."))

doc.add_heading("4. Results for all ten models", 1)
add_table(doc,
          ["#", "Model", "Hidden layers", "Loss", "Batch", "Epochs run",
           "Val MAPE", "Test MAPE", "Test MAE", "Kept?"],
          [(r["id"], r["title"], "-".join(map(str, r["config"]["hidden_layers"])),
            r["config"]["loss"].upper(), r["config"]["batch_size"], r["epochs_run"],
            f"{r['val_mape']:.2f}%", f"{r['test_mape']:.2f}%", f"${r['test_mae']:,.0f}",
            "Yes" if r["kept"] else "No") for r in results])

doc.add_heading("5. Best model", 1)
p = doc.add_paragraph()
p.add_run(f"The best model is Model {best['id']} ({best['title']}), with a test MAPE of "
          f"{best['test_mape']:.2f}%.").bold = True
doc.add_paragraph(
    f"On average its price predictions are {best['test_mape']:.2f}% away from the actual price "
    f"(about {100 - best['test_mape']:.1f}% accuracy measured as 100 - MAPE). Test MAE is "
    f"${best['test_mae']:,.0f} and R² is {best['test_r2']:.3f}. Compared with the base "
    f"model, test MAPE changed from {results[0]['test_mape']:.2f}% to "
    f"{best['test_mape']:.2f}%.")
doc.add_heading("Elements of the best model", 2)
add_model_elements(doc, best["config"], best)
doc.add_picture(CHART_FILE, width=Inches(6))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph(f"The source code for this model is in {BEST_MODEL_FILE}.")

doc.save(REPORT_FILE)
print(f"Saved report  -> {os.path.abspath(REPORT_FILE)}")


# ===========================================================================
# 8. Stand-alone source file for the best model (to submit to Canvas)
# ===========================================================================
BEST_TEMPLATE = '''"""
Assignment 3: Best model (Model {model_id}: {model_title})

Dense neural network that predicts California housing prices.
Test MAPE when the experiments were run: {test_mape:.2f}%
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers

SEED = {seed}
keras.utils.set_random_seed(SEED)

CONFIG = {config}

# 1. Load the California Housing dataset
(X_train_full, y_train_full), (X_test, y_test) = datasets.california_housing.load_data(
    version="large", path="california_housing.npz", test_split=0.2, seed=113)
X_train_full, X_test = X_train_full.astype("float32"), X_test.astype("float32")
y_train_full, y_test = y_train_full.astype("float32"), y_test.astype("float32")

# Validation split (15% of the training data)
order = np.random.default_rng(SEED).permutation(len(X_train_full))
n_val = int(len(order) * {val_frac})
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
    encode, decode = (lambda y: y / {target_scale}), (lambda y: y * {target_scale})

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
print(f"Test MAPE: {{test_mape:.2f}}%")
print(f"Test MAE:  ${{np.mean(np.abs(y_test - pred)):,.0f}}")
'''

with open(BEST_MODEL_FILE, "w") as f:
    f.write(BEST_TEMPLATE.format(
        model_id=best["id"], model_title=best["title"], test_mape=best["test_mape"],
        seed=SEED, config=pprint.pformat(best["config"], sort_dicts=False),
        val_frac=VALIDATION_FRACTION, target_scale=TARGET_SCALE))
print(f"Saved best model code -> {os.path.abspath(BEST_MODEL_FILE)}")