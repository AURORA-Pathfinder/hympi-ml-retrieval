import mlflow
import optuna
import numpy as np
import keras
import keras.backend
from keras import losses, metrics, callbacks, optimizers
from keras.layers import Dense, Concatenate, GaussianNoise
from keras import layers

from hympi_ml.utils import mlflow_log
from hympi_ml.data.fulldays import DKey, DPath, get_split_datasets
from hympi_ml.data.fulldays.preprocessing import get_minmax
from hympi_ml.layers import mlp, transform, loss
from hympi_ml.evaluation import log_eval_metrics, profile
from hympi_ml.utils.gpu import set_gpus


target_names = [DKey.TEMPERATURE, DKey.WATER_VAPOR]


def _objective(trial: optuna.Trial):
    set_gpus(min_free=0.8)
    keras.backend.clear_session()

    with mlflow.start_run(nested=True):
        features_names = [DKey.HA, DKey.HD, DKey.HW]
        # features_names = [DKey.ATMS]

        if trial.suggest_categorical("use_cpl", [True, False]):
            features_names.append(DKey.CPL)

        (train, validation, test) = get_split_datasets(
            loader_path=DPath.CPL_06,
            feature_names=features_names,
            target_names=target_names,
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

        activation = "gelu"  # trial.suggest_categorical("activation", ["relu"])
        mlflow.log_param("activation", activation)

        ## Path
        for name in features_names:
            input_layer = input_layers[name]
            out = input_layer

            (mins, maxs) = get_minmax(train.loader, name)
            out = transform.create_minmax(mins, maxs)(out)

            # if name != DKey.HW and name != DKey.ATMS:
            #     out = Dense(128, activation)(out)

            # out = Dense(128, activation)(out)
            out = Dense(64, activation)(out)

            output_layers.append(out)

        if len(output_layers) > 1:
            output = Concatenate()(output_layers)
        elif len(output_layers) == 1:
            output = output_layers[0]

        size = 256  # trial.suggest_categorical("size", [256])
        count = 2  # trial.suggest_categorical("count", [2])

        dropout_rate = 0.0  # trial.suggest_categorical("d_rate", [0.01, 0.1])
        mlflow.log_param("dropout_rate", dropout_rate)

        dense_layers = mlp.get_dense_layers(
            input_layer=output,
            sizes=[size] * count,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        output_layers = train.get_output_layers()

        for target in target_names:
            output_layers[target] = output_layers[target](dense_layers)

        model = keras.Model(list(input_layers.values()), list(output_layers.values()))

        # lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, amsgrad=True),
            loss=losses.MeanAbsolutePercentageError(),
            metrics=[metrics.MAE, metrics.MSE],
        )
        model.summary()

        # Training
        batch_size = 1024
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
                callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1),
                callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
            ],
        )

        # Evaluation
        profile.log_eval_profile_plots(model, train, test, count=100000)
        test_metrics = log_eval_metrics(model, test_batches, "test")

        return test_metrics["loss"]


with mlflow_log.start_run(
    tracking_uri="/data/nature_run/hympi-ml-retrieval/mlruns",
    experiment_name="+".join([key.name for key in target_names]),
    log_datasets=False,
):
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=4)
