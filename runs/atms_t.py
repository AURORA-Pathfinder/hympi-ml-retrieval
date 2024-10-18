import mlflow
import optuna
import numpy as np
import keras
import keras.backend
from keras import losses, metrics, callbacks, optimizers
from keras.layers import Dense, Concatenate

from hympi_ml.utils import mlflow_log
from hympi_ml.data.fulldays import DKey, get_split_datasets
from hympi_ml.data.fulldays.preprocessing import get_minmax
from hympi_ml.layers import mlp, transform
from hympi_ml.evaluation import log_eval_metrics, profile
from hympi_ml.utils.gpu import set_gpus


target_name = DKey.TEMPERATURE


def _objective(trial: optuna.Trial):
    set_gpus(min_free=0.8)
    keras.backend.clear_session()

    with mlflow.start_run(nested=True):
        use_cpl = trial.suggest_categorical("use_cpl", [True, False])
        use_spress = trial.suggest_categorical("use_spress", [True, False])

        features_names = [DKey.ATMS]

        if use_cpl:
            features_names.append(DKey.CPL)

        if use_spress:
            features_names.append(DKey.SURFACE_PRESSURE)

        (train, validation, test) = get_split_datasets(
            feature_names=features_names,
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
            autolog=True,
        )

        # Model Creation
        input_layers = train.get_input_layers()
        output_layers = []

        # ATMS Path
        atms_input = input_layers[DKey.ATMS]
        (mins, maxs) = get_minmax(train.loader, DKey.ATMS)
        atms_output = transform.create_minmax(mins, maxs)(atms_input)
        output_layers.append(atms_output)

        # CPL Path
        if use_cpl:
            cpl_input = input_layers[DKey.CPL]
            (mins, maxs) = get_minmax(train.loader, DKey.CPL)
            cpl_output = transform.create_minmax(mins, maxs)(cpl_input)
            cpl_output = Dense(128, "relu")(cpl_output)
            cpl_output = Dense(64, "relu")(cpl_output)
            output_layers.append(cpl_output)

        # Spress Path
        if use_spress:
            spress_input = input_layers[DKey.SURFACE_PRESSURE]
            (mins, maxs) = get_minmax(train.loader, DKey.SURFACE_PRESSURE)
            spress_output = transform.create_minmax(mins, maxs)(spress_input)
            output_layers.append(spress_output)

        output = Concatenate()(output_layers)

        size = trial.suggest_categorical("size", [128, 256])
        count = trial.suggest_categorical("count", [2, 4, 8])

        activation = trial.suggest_categorical("activation", ["gelu", "relu"])
        mlflow.log_param("activation", activation)

        dropout_rate = trial.suggest_categorical("d_rate", [0.0, 0.05, 0.1])
        mlflow.log_param("dropout_rate", dropout_rate)

        dense_layers = mlp.get_dense_layers(
            input_layer=output,
            sizes=[size] * count,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        output = Dense(72)(dense_layers)
        model = keras.Model(list(input_layers.values()), output)

        model.compile(optimizer=optimizers.Adam(), loss=losses.MAE, metrics=[metrics.MAE, metrics.MSE])
        model.summary()

        # Training
        batch_size = 500
        mlflow.log_param("memmap_batch_size", batch_size)

        train_batches = train.create_batches(batch_size)
        val_batches = validation.create_batches(batch_size)
        test_batches = test.create_batches(batch_size)

        model.fit(
            train_batches,
            validation_data=val_batches,
            epochs=1000,
            verbose=1,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
            ],
        )

        # Evaluation
        profile.log_eval_profile_plots(model, train, test, count=100000)
        test_metrics = log_eval_metrics(model, test_batches, "test")

        return test_metrics["loss"]


with mlflow_log.start_run(
    tracking_uri="/data/nature_run/hympi-ml-retrieval/mlruns",
    experiment_name=target_name.name,
    log_datasets=False,
):
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=10)
