# %%
import sys
sys.path.insert(0, '..')

import mlflow
import mlflow.keras
import numpy as np
import matplotlib.pyplot as plt

from keras import callbacks, activations, optimizers, losses, metrics
import keras.layers as layers
from keras.layers import Input, Dense, Dropout, Normalization, BatchNormalization
from keras.models import Model

from preprocessing.fulldays import FullDaysDataset, DataName
from model_creation.transform import create_norm_layer
from evaluation.metrics import log_eval_metrics
import evaluation.plots as plots

import utils.gpu as gpu
gpu.set_gpus(min_free=0.75)

import utils.mlflow_logging as mlflow_logging
mlflow_logging.start_run("HSEL Autoencoder", log_datasets=False)
# %%
# Loading

batch_size = 32
mlflow.log_param("memmap_batch_size", batch_size)

name = "HSEL_AE"

train = FullDaysDataset(days=[
    "20060315", "20060515", "20060615", "20060715", 
    "20060915", "20061015", "20061115", "20061215",
], feature_names=[DataName.hsel], target_name=DataName.hsel, name=name)

train_batches = train.create_batches(batch_size)
train.log("train")

validation = FullDaysDataset(days=["20060815"], feature_names=[DataName.hsel], target_name=DataName.hsel, name=name)
val_batches = validation.create_batches(batch_size)
validation.log("validation")

test = FullDaysDataset(days=["20060803"], feature_names=[DataName.hsel], target_name=DataName.hsel, name=name)
test_batches = test.create_batches(batch_size)
test.log("test")

# %%
# Model Creation

latent_size = 100
mlflow.log_param("latent_size", latent_size)

hsel_train = train.features[DataName.hsel.name]
hsel_test = test.features[DataName.hsel.name]

input_layer = Input(shape=hsel_train.shape[1:], name="hsel")
encoder = create_norm_layer(hsel_train, 10000)(input_layer)
encoder = (Dense(latent_size, activations.gelu)(encoder))
decoder = (Dense(hsel_train.data_shape[0], activations.linear)(encoder))
autoencoder = Model(input_layer, decoder)
encoder = Model(input_layer, encoder)

autoencoder.compile(optimizer=optimizers.Adam(), loss=losses.Huber(), metrics=[metrics.MAE, metrics.MSE])
autoencoder.summary()

# %%
# Training

early_stopping = callbacks.EarlyStopping(patience=10, verbose=1)
autoencoder.fit(train_batches, validation_data=val_batches, epochs=500, verbose=1, callbacks=[early_stopping])

# %%
# Evaluation

enc_path = f"encoder.keras"
encoder.save(f"/tmp/{enc_path}")
mlflow.log_artifact(f"/tmp/{enc_path}")

log_eval_metrics(autoencoder, test_batches, "test")

print("TEST DATASET")
for i in range(5):
    fig = plots.plot_pred_error(autoencoder, hsel_test, hsel_test, y_label="Radiance (K)")
    mlflow.log_figure(fig, f"test_pred_error_{i}.png")

print("TRAIN DATASET")
for i in range(5):
    fig = plots.plot_pred_error(autoencoder, hsel_train, hsel_train, y_label="Radiance (K)")
    mlflow.log_figure(fig, f"train_pred_error_{i}.png")

mlflow.end_run()