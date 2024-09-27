# %% Imports and Initialization
import sys

sys.path.insert(0, "..")

import mlflow
import mlflow.keras
import numpy as np
import matplotlib.pyplot as plt

import keras.callbacks as callbacks
import keras.activations as act
import keras.optimizers as opt
import keras.losses as losses
import keras.metrics as metrics
from keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Concatenate,
    LayerNormalization,
)
from keras.models import Model

import utils.mlflow_logging as mlflow_logging
import data.fulldays as fd
from data.fulldays.loading import DKey

from evaluation.metrics import log_eval_metrics
import evaluation.plots as plots

from utils.gpu import set_gpus

set_gpus(min_free=0.75)


# %% Data Loading
instr = DKey.HSEL
target_name = DKey.TEMPERATURE

mlflow_logging.start_run(
    tracking_uri="/data/nature_run/hympi-ml-retrieval/mlruns",
    experiment_name=target_name.name,
    log_datasets=False,
)

(train, validation, test) = fd.get_split_datasets(
    feature_names=[instr, DKey.SURFACE_PRESSURE],
    target_name=target_name,
    train_days=[
        "20060315",
        "20060515",
        "20060615",
        "20060715",
        "20060915",
        "20061015",
        "20061115",
        "20061215",
    ],
    validation_days=["20060815"],
    test_days=["20060803"],
    logging=True,
)

# %% Model Creation
input_layers = train.get_input_layers()

instr_input = input_layers[instr]
instr_output = instr_input

# encoder = mlflow_logging.get_model("6d1374316f334f65a6d529d2ddcaf4f8", "encoder.keras")
# encoder.trainable = False
# instr_output = encoder(instr_output)
instr_output = LayerNormalization()(instr_output)

spress_input = input_layers[DKey.SURFACE_PRESSURE]
spress_output = LayerNormalization()(spress_input)

activation = act.swish
size = 128
output = Concatenate()([instr_output, spress_output])
output = Dense(size, activation)(output)
output = Dense(size, activation)(output)
output = Dense(size, activation)(output)
output = Dense(72)(output)
model = Model([instr_input, spress_input], output)

model.compile(optimizer=opt.Adam(), loss=losses.MAE, metrics=[metrics.MAE, metrics.MSE])

model.summary()

# %% Training

batch_size = 2000
mlflow.log_param("memmap_batch_size", batch_size)

train_batches = train.create_batches(batch_size)
val_batches = validation.create_batches(batch_size)
test_batches = test.create_batches(batch_size)

early_stopping = callbacks.EarlyStopping(patience=10, verbose=1)
model.fit(
    train_batches,
    validation_data=val_batches,
    epochs=1000,
    verbose=1,
    callbacks=[early_stopping],
)

# %% Evaluation

log_eval_metrics(model, test_batches, "test")

steps = 10
test_batches.shuffle = False
test_pred = model.predict(test_batches, steps=steps)
test_truth = test.target[0 : steps * batch_size]

var = plots.plot_pred_truth_var(
    test_pred, test_truth, context=f"{steps} Test Batches", show=True
)
mlflow.log_figure(var, var.get_suptitle() + ".png")

mae = plots.plot_mae_per_level(
    test_pred, test_truth, context=f"{steps} Test Batches", show=True
)
mlflow.log_figure(mae, mae.get_suptitle() + ".png")

train_batches.shuffle = False
train_pred = model.predict(train_batches, steps=steps)
train_truth = train.target[0 : steps * batch_size]

var = plots.plot_pred_truth_var(
    train_pred, train_truth, context=f"{steps} Train Batches", show=True
)
mlflow.log_figure(var, var.get_suptitle() + ".png")

mae = plots.plot_mae_per_level(
    train_pred, train_truth, context=f"{steps} Train Batches", show=True
)
mlflow.log_figure(mae, mae.get_suptitle() + ".png")

print("TEST DATASET")
for i in range(3):
    fig = plots.plot_pred_error(
        model, data=test, x_label="Sigma", y_label="Temperature (K)"
    )
    mlflow.log_figure(fig, f"test_pred_error_{i}.png")

print("TRAIN DATASET")
for i in range(3):
    fig = plots.plot_pred_error(
        model, data=train, x_label="Sigma", y_label="Temperature (K)"
    )
    mlflow.log_figure(fig, f"train_pred_error_{i}.png")

mlflow.end_run()
