import gc

import mlflow
import keras
from keras import losses, metrics, callbacks, optimizers, layers
import tensorflow as tf

from hympi_ml.utils import mlflow_log
from hympi_ml.data.model_dataset import get_split_datasets, get_datasets_from_run
from hympi_ml.data import DataSpec, DataLoader, RFBand

# from hympi_ml.data.dataspec import DataSpec
# from hympi_ml.data.rfband import RFBand
from hympi_ml.data.hympi import HympiSpec
from hympi_ml.data.nature_run import NRSpec
from hympi_ml.data.allsky import AllSkyLoader
from hympi_ml.layers import mlp
from hympi_ml.evaluation import figs
from hympi_ml.utils.gpu import set_gpus

from rich import traceback

traceback.install()


def start_run(
    features: list[DataSpec],
    targets: list[DataSpec],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
):
    set_gpus(min_free=0.8)
    keras.backend.clear_session()
    gc.collect()

    with mlflow.start_run(nested=True):
        (train, validation, test) = get_split_datasets(
            features,
            targets,
            train_loader,
            validation_loader,
            test_loader,
            autolog=True,
        )

        # Model Creation
        input_layers = train.get_input_layers()
        output_layers = []

        activation = "gelu"
        mlflow.log_param("activation", activation)

        ## Path
        for name in train._feature_names:
            input_layer = input_layers[name]
            out = input_layer

            size = train.feature_shapes[name][0]

            out = layers.LayerNormalization()(out)

            if size > 32:
                # out = layers.Dense(int(size / 4), activation)(out)
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
            sizes=[128, 128],
            activation=activation,
            dropout_rate=dropout_rate,
        )

        output_layers = train.get_output_layers()

        for target in train._target_names:
            output_layers[target] = output_layers[target](dense_layers)

        model = keras.Model(input_layers, output_layers)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, amsgrad=True),
            loss=losses.MeanAbsoluteError(),
            metrics=[metrics.MeanAbsoluteError(), metrics.MeanAbsoluteError()],
        )
        model.summary()

        # Training
        batch_size = 512
        mlflow.log_param("data_batch_size", batch_size)

        train_ds = (
            train.as_tf_dataset()
            .cache()
            .shuffle(buffer_size=2**18, reshuffle_each_iteration=True)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = (
            validation.as_tf_dataset()
            .batch(batch_size)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1000,
            verbose=1,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=6, verbose=1, min_delta=0.0001
                ),
                callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6),
            ],
        )

        # Evaluation
        figs.log_metric_profile_figs(
            mlflow.active_run().info.run_id,
            [
                figs.MeanErrorProfile(absolute=False, percentage=False),
                figs.MeanErrorProfile(absolute=True, percentage=False),
                figs.MeanErrorProfile(absolute=False, percentage=True),
                figs.MeanErrorProfile(absolute=True, percentage=True),
                figs.VarCompProfile(),
            ],
        )
        test.evaluate(
            model, metrics=["mae", "mse"], context="test", unscale=True, log=True
        )


AA_THRESHOLD = [
    RFBand(low=118, high=126),
    RFBand(low=126, high=166, resolution=0.5),
    RFBand(low=177, high=183),
]

AA_BASELINE = [
    RFBand(low=114, high=126),
    RFBand(low=125, high=175, resolution=0.5),
    RFBand(low=171, high=183),
]

AA_ASPIRATIONAL1 = [
    RFBand(low=114, high=126),
    RFBand(low=125, high=175, resolution=0.5),
    RFBand(low=174, high=186),
]

AA_ASPIRATIONAL2 = [
    RFBand(low=114, high=126),
    RFBand(low=125, high=175, resolution=0.5),
    RFBand(low=174, high=192),
]

# with mlflow_log.start_run(
#     tracking_uri="/home/dgershm1/mlruns",
#     experiment_name="tq",
#     log_datasets=False,
# ):
#     start_run(
#         features=[
#             HympiSpec(name="AURORA_THRESHOLD", freqs=AA_THRESHOLD, nedt="AURORA"),
#         ],
#         targets=[
#             NRSpec(name="TEMPERATURE", dataset="TEMPERATURE", scale_range=(175, 325)),
#             NRSpec(name="WATER_VAPOR", dataset="WATER_VAPOR", scale_range=(1.11e-7, 3.08e-2)),
#         ],
#         train_loader=AllSkyLoader(
#             days=[
#                 "20060315",
#                 "20060515",
#                 "20060615",
#                 "20060715",
#                 "20060915",
#                 "20061015",
#                 "20061115",
#                 "20061215",
#                 "20060815",
#             ],
#         ),
#         validation_loader=AllSkyLoader(days=["20060803"]),
#         test_loader=AllSkyLoader(days=["20060803"]),
#     )

mlflow.set_tracking_uri("/home/dgershm1/mlruns")

with mlflow.start_run(run_id="7c12507b4ebe4be49c1396628003b3ee"):
    model = mlflow_log.get_autolog_model("7c12507b4ebe4be49c1396628003b3ee")
    dataset_dict = get_datasets_from_run("7c12507b4ebe4be49c1396628003b3ee")
    dataset_dict["test"].evaluate(
        model, metrics=["mae", "mse"], context="test", unscale=True, log=True
    )
