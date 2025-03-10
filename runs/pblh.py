import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is
import gc

import mlflow
import keras
from keras import losses, metrics, callbacks, optimizers, layers
import tensorflow as tf

from hympi_ml.utils import mlflow_log
from hympi_ml.data.fulldays import DKey, DPath, get_split_datasets
from hympi_ml.layers import mlp, noise
from hympi_ml.evaluation import figs
from hympi_ml.utils.gpu import set_gpus

target_names = [DKey.PBLH]

from rich import traceback
traceback.install()


def start_run(feature_names: list[DKey], add_nedt: bool):
    set_gpus(min_free=0.8)
    keras.backend.clear_session()
    gc.collect()

    with mlflow.start_run(nested=True):
        (train, validation, test) = get_split_datasets(
            data_path=DPath.CPL_06,
            train_days=[
                "20060315",
                # "20060515",
                # "20060615",
                # "20060715",
                # "20060915",
                # "20061015",
                # "20061115",
                # "20061215",
                # "20060815",
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

        mlflow.log_param("add_nedt", add_nedt)

        ## Path
        for name in feature_names:
            input_layer = input_layers[name]
            out = input_layer

            size = train.feature_shapes[name][0]

            if add_nedt:
                out = noise.add_nedt_layer(out, name)

            out = layers.LayerNormalization()(out)

            if size > 32:
                out = layers.Dense(int(size / 4), activation)(out)
                out = layers.Dense(int(size / 8), activation)(out)

            output_layers.append(out)

        if len(output_layers) > 1:
            output = layers.Concatenate()(output_layers)
        else:
            output = output_layers[0]

        dropout_rate = 0.0
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

        model = keras.Model(input_layers, output_layers)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, amsgrad=True),
            loss=losses.MeanAbsoluteError(),
            metrics=[metrics.MeanAbsoluteError()],
        )
        model.summary()

        # Training
        batch_size = 2048
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
            epochs=1,
            verbose=1,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=8, verbose=1),
                callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6),
            ],
        )

        # Evaluation
        figs.log_metric_profile_figs(mlflow.active_run().info.run_id, [figs.MeanErrorHistogram()])
        test.evaluate(model, metrics=["mae", "mse"], context="test", unscale=True, log=True)


with mlflow_log.start_run(
    tracking_uri="/home/dgershm1/mlruns",
    experiment_name="+".join([key.name for key in target_names]),
    log_datasets=False,
):
    # start_run(feature_names=[DKey.ATMS], add_nedt=True)
    # start_run(feature_names=[DKey.ATMS, DKey.CPL], add_nedt=True)
    # start_run(feature_names=[DKey.HA, DKey.HD, DKey.HW], add_nedt=True)
    start_run(feature_names=[DKey.HA, DKey.HD, DKey.HW, DKey.CPL], add_nedt=True)
