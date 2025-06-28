import os

os.environ["KERAS_BACKEND"] = "torch"

import gc
from typing import Mapping

import torch
import mlflow
import keras
from keras import losses, metrics, callbacks, optimizers, layers

from torch.utils.data import DataLoader

from hympi_ml.utils import mlflow_log
from hympi_ml.data.model_dataset import get_split_datasets

from hympi_ml.data import (
    DataSpec,
    DataSource,
    CosmirhSpec,
    AMPRSpec,
    NRSpec,
    ch06,
    RFBand,
)
from hympi_ml.data.cosmirh import C1_band, C4_band
from hympi_ml.layers import mlp
from hympi_ml.evaluation.metrics import ErrorHistogram

from rich import traceback

traceback.install()


def start_run(
    features: Mapping[str, DataSpec],
    targets: Mapping[str, DataSpec],
    train_source: DataSource,
    validation_source: DataSource,
    test_source: DataSource,
):
    keras.backend.clear_session()
    gc.collect()

    with mlflow.start_run(nested=True):
        (train, validation, test) = get_split_datasets(
            features,
            targets,
            train_source,
            validation_source,
            test_source,
            autolog=True,
        )

        # Model Creation
        inputs = train.get_input_layers()
        outputs = []

        activation = "gelu"
        mlflow.log_param("activation", activation)

        ## Path
        for name in train.feature_names:
            input_layer = inputs[name]
            out = input_layer

            size = train.feature_shapes[name][0]

            if size > 32:
                out = layers.Dense(int(size / 4), activation)(out)

            outputs.append(out)

        if len(outputs) > 1:
            output = layers.Concatenate()(outputs)
        else:
            output = outputs[0]

        dropout_rate = 0.0
        mlflow.log_param("dropout_rate", dropout_rate)

        out_dense = mlp.get_dense_layers(
            input_layer=output,
            sizes=[128, 32],
            activation=activation,
            dropout_rate=dropout_rate,
        )

        outputs = {k: (v)(out_dense) for k, v in train.get_output_layers().items()}

        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, amsgrad=True),
            loss=losses.MeanAbsoluteError(),
            metrics=[metrics.MeanAbsoluteError(), metrics.MeanAbsoluteError()],
        )
        model.summary()

        # Training
        batch_size = 2048
        mlflow.log_param("data_batch_size", batch_size)

        train.batch_size = batch_size
        validation.batch_size = 16384
        test.batch_size = 16384

        train_ds = DataLoader(
            train,
            batch_size=None,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

        val_ds = DataLoader(
            validation,
            batch_size=None,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=6,
            verbose=1,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=6, verbose=1, min_delta=0.0001
                ),
                callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6),
            ],
        )

        # Evaluation
        test.evaluate(
            model,
            metrics={
                "PBLH": [ErrorHistogram(bins=200, min=-500, max=500)],
            },
            context="test",
            unscale=True,
            log=True,
        )


with mlflow_log.start_run(
    tracking_uri="/home/dgershm1/mlruns",
    experiment_name="pblh",
    log_datasets=False,
):
    start_run(
        features={
            "CH_8": CosmirhSpec(
                frequencies=[
                    RFBand(low=C1_band.low, high=58, resolution=2 ** (-5)),
                    RFBand(low=C4_band.low, high=192, resolution=2 ** (-5)),
                ],
                ignore_frequencies=[  # problematic CRTM frequencies
                    56.96676875,
                    57.60739375,
                    57.6113,
                    57.61520625,
                ],
                scale_range=(200, 300),
            ),
        },
        targets={"PBLH": NRSpec(dataset="PBLH", scale_range=(200, 2000))},
        train_source=ch06.Ch06Source(
            days=[
                "20060115",
                "20060215",
                "20060315",
                "20060415",
                "20060515",
                "20060615",
                "20060715",
            ],
        ),
        validation_source=ch06.Ch06Source(days=["20060815"]),
        test_source=ch06.Ch06Source(days=["20060815"]),
    )
