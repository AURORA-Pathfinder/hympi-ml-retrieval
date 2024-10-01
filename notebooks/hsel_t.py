import sys

sys.path.insert(0, "..")

import mlflow.keras
import mlflow
import optuna

import keras.callbacks as callbacks
import keras.activations as act
import keras.optimizers as opt
import keras.losses as losses
import keras.metrics as metrics
from keras.layers import (
    Dense,
    Concatenate,
    LayerNormalization,
)
from keras.backend import clear_session
from keras.models import Model

import utils.mlflow_logging as mlflow_logging
import data.fulldays as fd
from data.fulldays.loading import DKey

from evaluation.metrics import log_eval_metrics
import evaluation.plots as plots

from utils.gpu import set_gpus


target_name = DKey.TEMPERATURE


def _objective(trial: optuna.Trial):
    set_gpus(min_free=0.99)
    clear_session()

    with mlflow.start_run(nested=True):
        (train, validation, test) = fd.get_split_datasets(
            feature_names=[DKey.HSEL, DKey.SURFACE_PRESSURE],
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
                "20060815",
            ],
            validation_days=["20060803"],
            test_days=["20060803"],
        )

        train.log("train")
        validation.log("validation")
        test.log("test")

        # Model Creation
        input_layers = train.get_input_layers()

        hsel_input = input_layers[DKey.HSEL]
        hsel_output = hsel_input

        encoder = mlflow_logging.get_model("6d1374316f334f65a6d529d2ddcaf4f8", "encoder.keras")
        encoder.trainable = False
        hsel_output = encoder(hsel_output)
        hsel_output = LayerNormalization()(hsel_output)

        spress_input = input_layers[DKey.SURFACE_PRESSURE]
        spress_output = LayerNormalization()(spress_input)

        size = trial.suggest_categorical("size", [64, 128, 256, 512])
        activation = trial.suggest_categorical("activation", ["swish", "gelu", "relu"])

        output = Concatenate()([hsel_output, spress_output])
        output = Dense(size, activation)(output)
        output = Dense(size, activation)(output)
        output = Dense(size, activation)(output)
        output = Dense(72)(output)
        model = Model([hsel_input, spress_input], output)

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        model.compile(
            optimizer=opt.Adam(learning_rate=lr),
            loss=losses.MAE,
            metrics=[metrics.MAE, metrics.MSE],
        )

        model.summary()

        # Training

        batch_size = 2048
        mlflow.log_param("memmap_batch_size", batch_size)

        train_batches = train.create_batches(batch_size)
        val_batches = validation.create_batches(batch_size)
        test_batches = test.create_batches(batch_size)

        early_stopping = callbacks.EarlyStopping(patience=3, verbose=1)
        model.fit(
            train_batches,
            validation_data=val_batches,
            epochs=1000,
            verbose=2,
            callbacks=[early_stopping],
        )

        # Evaluation

        log_eval_metrics(model, test_batches, "test")

        steps = 10
        test_batches.shuffle = False
        test_pred = model.predict(test_batches, steps=steps)
        test_truth = test.target[0 : steps * batch_size]

        var = plots.plot_pred_truth_var(test_pred, test_truth, context=f"{steps} Test Batches", show=False)
        mlflow.log_figure(var, var.get_suptitle() + ".png")

        mae = plots.plot_mae_per_level(test_pred, test_truth, context=f"{steps} Test Batches", show=False)
        mlflow.log_figure(mae, mae.get_suptitle() + ".png")

        train_batches.shuffle = False
        train_pred = model.predict(train_batches, steps=steps)
        train_truth = train.target[0 : steps * batch_size]

        var = plots.plot_pred_truth_var(train_pred, train_truth, context=f"{steps} Train Batches", show=False)
        mlflow.log_figure(var, var.get_suptitle() + ".png")

        mae = plots.plot_mae_per_level(train_pred, train_truth, context=f"{steps} Train Batches", show=False)
        mlflow.log_figure(mae, mae.get_suptitle() + ".png")

        print("TEST DATASET")
        for i in range(3):
            fig = plots.plot_pred_error(model, data=test, x_label="Sigma", y_label="Temperature (K)")
            mlflow.log_figure(fig, f"test_pred_error_{i}.png")

        print("TRAIN DATASET")
        for i in range(3):
            fig = plots.plot_pred_error(model, data=train, x_label="Sigma", y_label="Temperature (K)")
            mlflow.log_figure(fig, f"train_pred_error_{i}.png")


with mlflow_logging.start_run(
    tracking_uri="/data/nature_run/hympi-ml-retrieval/mlruns",
    experiment_name=target_name.name,
    log_datasets=False,
):
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=20)
