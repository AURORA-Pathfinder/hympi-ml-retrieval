# %% Imports and Initialization
import sys

sys.path.insert(0, "..")

import mlflow
import mlflow.keras

import keras.callbacks as callbacks
import keras.activations as act
import keras.optimizers as opt
import keras.losses as losses
import keras.metrics as metrics
from keras.layers import Dense, Dropout
from keras.models import Model

import utils.mlflow_logging as mlflow_logging
from data.fulldays.loading import DKey
from data.fulldays.dataset import get_split_datasets
from model_creation.transform import create_norm_layer
from evaluation.metrics import log_eval_metrics
import evaluation.plots as plots

from utils.gpu import set_gpus

set_gpus(min_free=0.75)

current_run = mlflow_logging.start_run("HSEL Autoencoder", log_datasets=False)

# %% Data Loading
(train, validation, test) = get_split_datasets(
    feature_names=[DKey.HSEL],
    target_name=DKey.HSEL,
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
hsel_train = train.features[DKey.HSEL]

latent_size = 100
mlflow.log_param("latent_size", latent_size)

input_layer = train.get_input_layers()[DKey.HSEL]
encoder = create_norm_layer(hsel_train, 25000)(input_layer)
encoder = Dense(latent_size, act.gelu)(encoder)
decoder = Dense(hsel_train.data_shape[0], act.linear)(encoder)
autoencoder = Model(input_layer, decoder)
encoder = Model(input_layer, encoder)

autoencoder.compile(optimizer=opt.Adam(), loss=losses.Huber(), metrics=[metrics.MAE, metrics.MSE])
autoencoder.summary()

# %% Training

batch_size = 64
mlflow.log_param("memmap_batch_size", batch_size)

train_batches = train.create_batches(batch_size)
val_batches = validation.create_batches(batch_size)
test_batches = test.create_batches(batch_size)

early_stopping = callbacks.EarlyStopping(patience=5, verbose=1)
autoencoder.fit(
    train_batches,
    validation_data=val_batches,
    epochs=500,
    verbose=1,
    callbacks=[early_stopping],
)

enc_path = "encoder.keras"
encoder.save(f"/tmp/{enc_path}")
mlflow.log_artifact(f"/tmp/{enc_path}")

# %% Evaluation
log_eval_metrics(autoencoder, test_batches, "test")

print("TEST DATASET")
for i in range(5):
    fig = plots.plot_pred_error(autoencoder, data=test, y_label="Radiance (K)")
    mlflow.log_figure(fig, f"test_pred_error_{i}.png")

print("TRAIN DATASET")
for i in range(5):
    fig = plots.plot_pred_error(autoencoder, data=train, y_label="Radiance (K)")
    mlflow.log_figure(fig, f"train_pred_error_{i}.png")

mlflow.end_run()
