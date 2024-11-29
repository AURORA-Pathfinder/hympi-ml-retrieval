import gc

import mlflow
import optuna
import numpy as np
import keras
import keras.backend
from keras import losses, metrics, callbacks, optimizers

from hympi_ml.utils import mlflow_log
from hympi_ml.data.fulldays import DKey, get_split_datasets
from hympi_ml.data.fulldays.preprocessing import get_minmax
from hympi_ml.layers import mlp, transform
from hympi_ml.evaluation import log_eval_metrics, profile
from hympi_ml.utils.gpu import set_gpus


def _objective(trial: optuna.Trial):
    set_gpus(min_free=0.8)
    keras.backend.clear_session()
    gc.collect()

    with mlflow.start_run(nested=True):
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
                "20060815",
            ],
            validation_days=["20060803"],
            test_days=["20060803"],
            autolog=True,
        )

        # Model Creation
        input_layers = train.get_input_layers()

        hsel_input = input_layers[DKey.HSEL]
        hsel_output = hsel_input

        (mins, maxs) = get_minmax(train.loader, DKey.HSEL)
        hsel_output = transform.create_minmax(mins, maxs)(hsel_output)

        size = trial.suggest_categorical("size", [128, 256, 512])
        count = trial.suggest_categorical("count", [0, 1, 2, 3])

        latent_size = 500  # trial.suggest_categorical("latent_size", [32, 64, 128, 256])
        mlflow.log_param("latent_size", latent_size)

        activation = trial.suggest_categorical("activation", ["gelu", "relu"])
        mlflow.log_param("activation", activation)

        dropout_rate = trial.suggest_categorical("d_rate", [0.0, 0.05, 0.1])
        mlflow.log_param("dropout_rate", dropout_rate)

        (ae_layers, enc_layers) = mlp.get_autoencoder_layers(
            input_layer=hsel_output,
            input_size=train.features[DKey.HSEL].data_shape[0],
            encode_sizes=[size] * count,
            latent_size=latent_size,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        model = keras.Model(input_layers, ae_layers)
        encoder = keras.Model(input_layers, enc_layers)

        model.compile(optimizer=optimizers.Adam(), loss=losses.MAE, metrics=[metrics.MAE, metrics.MSE])
        model.summary()

        # Training
        batch_size = 2000

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

        enc_path = "encoder.keras"
        encoder.save(f"/tmp/{enc_path}")
        mlflow.log_artifact(f"/tmp/{enc_path}")

        # Evaluation
        profile.log_eval_profile_plots(model, train, test, count=10000)
        test_metrics = log_eval_metrics(model, test_batches, "test")

        return test_metrics["loss"]


with mlflow_log.start_run(
    tracking_uri="/data/nature_run/hympi-ml-retrieval/mlruns",
    experiment_name="HSEL Autoencoder",
    log_datasets=False,
):
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=5)
