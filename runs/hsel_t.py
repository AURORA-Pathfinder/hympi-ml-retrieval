import mlflow
import optuna
import keras
import keras.backend
from keras import losses, metrics, callbacks, optimizers
from keras.layers import Dense, Concatenate, LayerNormalization

from hympi_ml.utils import mlflow_log
from hympi_ml.data.fulldays import DKey, get_split_datasets
from hympi_ml.layers import mlp, transform
from hympi_ml.evaluation import log_eval_metrics, profile
from hympi_ml.utils.gpu import set_gpus


target_name = DKey.TEMPERATURE


def _objective(trial: optuna.Trial):
    set_gpus(min_free=0.8)
    keras.backend.clear_session()

    with mlflow.start_run(nested=True):
        (train, validation, test) = get_split_datasets(
            feature_names=[DKey.HSEL, DKey.CPL, DKey.SURFACE_PRESSURE],
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

        hsel_input = input_layers[DKey.HSEL]
        hsel_output = hsel_input

        encoder = mlflow_log.get_model("80ffa7825f2a47feab265a6729f6fa5b", "encoder.keras")
        encoder.trainable = False
        hsel_output = encoder(hsel_output)
        hsel_train = train.features[DKey.HSEL]
        hsel_output = transform.create_minmax_layer(hsel_train, 100_000)(hsel_output)
        hsel_output = Dense(64, "gelu")(hsel_output)

        cpl_input = input_layers[DKey.CPL]
        cpl_train = train.features[DKey.CPL]
        cpl_output = transform.create_minmax_layer(cpl_train, 100_000)(cpl_input)
        cpl_output = Dense(128, "gelu")(cpl_output)
        cpl_output = Dense(32, "gelu")(cpl_output)

        spress_input = input_layers[DKey.SURFACE_PRESSURE]
        spress_train = train.features[DKey.SURFACE_PRESSURE]
        spress_output = transform.create_minmax_layer(spress_train, 100_000)(spress_input)

        output = Concatenate()([hsel_output, cpl_output, spress_output])

        size = trial.suggest_categorical("size", [128, 256, 512])
        count = trial.suggest_categorical("count", [2, 4, 8])

        activation = trial.suggest_categorical("activation", ["gelu", "relu"])
        mlflow.log_param("activation", activation)

        dropout_rate = trial.suggest_categorical("d_rate", [0.0, 0.05, 0.1, 0.2])
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

        # Evaluation
        profile.log_eval_profile_plots(model, train, test, count=10000)
        test_metrics = log_eval_metrics(model, test_batches, "test")

        return test_metrics["loss"]


with mlflow_log.start_run(
    tracking_uri="/data/nature_run/hympi-ml-retrieval/mlruns",
    experiment_name=target_name.name,
    log_datasets=False,
):
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=10)
