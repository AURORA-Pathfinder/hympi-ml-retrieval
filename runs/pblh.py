import gc

import mlflow
import optuna
import numpy as np
import keras
import keras.backend
from keras import losses, metrics, callbacks, optimizers
from keras.layers import Dense, Concatenate, LayerNormalization
import tensorflow as tf

from hympi_ml.utils import mlflow_log
from hympi_ml.data.fulldays import DKey, DPath, get_split_datasets
from hympi_ml.layers import mlp, noise
from hympi_ml.evaluation import figs
from hympi_ml.utils.gpu import set_gpus

target_names = [DKey.PBLH]


def _objective(trial: optuna.Trial):
    set_gpus(min_free=0.8)
    keras.backend.clear_session()
    gc.collect()

    with mlflow.start_run(nested=True):
        feature_names = [DKey.HA, DKey.HD, DKey.HW]
        # feature_names = [DKey.ATMS]

        if trial.suggest_categorical("use_cpl", [False]):
            feature_names.append(DKey.CPL)

        (train, validation, test) = get_split_datasets(
            data_path=DPath.CPL_06,
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
            feature_names=feature_names,
            target_names=target_names,
            filters={
                DKey.PBLH: [(200, 2000)],
            },
            scale_ranges={
                DKey.PBLH: (200, 2000),
            },
            autolog=True,
        )

        # Model Creation
        input_layers = train.get_input_layers()
        output_layers = []

        activation = "gelu"
        mlflow.log_param("activation", activation)

        ## Path
        for name in feature_names:
            input_layer = input_layers[name]
            out = input_layer

            size = train.feature_shapes[name][0]

            out = noise.add_nedt_layer(out, name)

            out = LayerNormalization()(out)

            if size > 32:
                out = Dense(size / 8, activation)(out)
                # out = Dense(size / 4, activation)(out)
                # out = Dense(size / 8, activation)(out)

            output_layers.append(out)

        if len(output_layers) > 1:
            output = Concatenate()(output_layers)
        else:
            output = output_layers[0]

        dropout_rate = 0.0  # trial.suggest_categorical("d_rate", [0.0, 0.1])
        mlflow.log_param("dropout_rate", dropout_rate)

        dense_layers = mlp.get_dense_layers(
            input_layer=output,
            sizes=[128, 64, 32],
            activation=activation,
            dropout_rate=dropout_rate,
        )

        output_layers = train.get_output_layers()

        for target in target_names:
            output_layers[target] = output_layers[target](dense_layers)

        model = keras.Model(list(input_layers.values()), list(output_layers.values()))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, amsgrad=True),
            loss=losses.MAE,
            metrics=[metrics.MAE],
        )
        model.summary()

        # Training
        batch_size = 1024
        mlflow.log_param("data_batch_size", batch_size)

        train_ds = (
            train.as_tf_dataset()
            .cache()
            .shuffle(buffer_size=2**18, reshuffle_each_iteration=True)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = validation.as_tf_dataset().batch(batch_size).prefetch(tf.data.AUTOTUNE)

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1000,
            verbose=1,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
                callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
            ],
        )

        # Evaluation
        figs.log_metric_profile_figs(mlflow.active_run().info.run_id, [figs.MeanErrorHistogram()])
        test_metrics = test.evaluate(model, metrics=["mae", "mse"], context="test", unscale=True, log=True)
        return list(test_metrics.values())[0]


with mlflow_log.start_run(
    tracking_uri="/data/nature_run/hympi-ml-retrieval/mlruns",
    experiment_name="+".join([key.name for key in target_names]),
    log_datasets=False,
):
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=1)
