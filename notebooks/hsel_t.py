# %%
import sys
sys.path.insert(0, '..')

import mlflow
import mlflow.keras
import numpy as np
import matplotlib.pyplot as plt

import keras.callbacks as callbacks
import keras.activations as act
import keras.optimizers as opt
import keras.losses as losses
import keras.metrics as metrics
from keras.layers import Input, Dense, Dropout, Normalization, BatchNormalization, Concatenate
from keras.models import Model

import utils.mlflow_logging as mlflow_logging
from preprocessing.fulldays import FullDaysDataset, DataName
from model_creation.encode import Encode
from model_creation.transform import create_norm_layer
from evaluation.metrics import log_eval_metrics
import evaluation.plots as plots

from utils.gpu import set_gpus
set_gpus(min_free=0.75)

current_run = mlflow_logging.start_run("Temperature", log_datasets=False)

# %%
batch_size = 1024
features = [DataName.hsel, DataName.surface_pressure]
target = DataName.temperature

train = FullDaysDataset(days=[
    "20060315", "20060515", "20060615", "20060715", 
    "20060915", "20061015", "20061115", "20061215",
], feature_names=features, target_name=target)

train_batches = train.create_batches(batch_size)
train.log("train")

validation = FullDaysDataset(days=["20060815"], feature_names=features, target_name=target)

val_batches = validation.create_batches(batch_size)
validation.log("validation")

test = FullDaysDataset(days=["20060803"], feature_names=features, target_name=target)

test_batches = test.create_batches(batch_size)
test.log("test")

# %%
hsel_train = train.features[DataName.hsel.name]
spress_train = train.features[DataName.surface_pressure.name]

input_layers = train.get_input_layers()

hsel_input = input_layers[DataName.hsel.name]
hsel_output = hsel_input
# hsel_output = Encode.from_mlflow_artifact("HSEL Autoencoder", "able-loon-114", "encoder.keras")(hsel_output)
hsel_output = create_norm_layer(hsel_train, 25000)(hsel_output)

spress_input = input_layers[DataName.surface_pressure.name]
spress_output = create_norm_layer(spress_train, 25000)(spress_input)

output = Concatenate()([hsel_output, spress_output])

output = (Dense(512, act.gelu)(output))
output = (Dense(128, act.gelu)(output))
output = (Dense(72, act.linear)(output))
model = Model([hsel_input, spress_input], output)

model.compile(optimizer=opt.Adam(), 
              loss=losses.MAE, 
              metrics=[metrics.MAE, metrics.MSE])

model.summary()

early_stopping = callbacks.EarlyStopping(patience=5, verbose=1)
model.fit(train_batches, validation_data=val_batches, epochs=500, verbose=1, callbacks=[early_stopping])

# %%
# plots.plot_variance_per_level(model.predict(train_batches, steps=50), show=True)
log_eval_metrics(model, test_batches, "test")

print("TEST DATASET")
for i in range(5):
    fig = plots.plot_pred_error(model, data=test, x_label="Level", y_label="Water Vapor (q)")
    mlflow.log_figure(fig, f"test_pred_error_{i}.png")

print("TRAIN DATASET")
for i in range(5):
    fig = plots.plot_pred_error(model, data=train, x_label="Level", y_label="Water Vapor (q)")
    mlflow.log_figure(fig, f"train_pred_error_{i}.png")


mlflow.end_run()