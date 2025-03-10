from typing import Any
import json
import hashlib

import tensorflow as tf
import keras
import numpy as np

import mlflow
import mlflow.types
from mlflow.data.dataset import Dataset, DatasetSource
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.tracking.context import registry
from mlflow.data.schema import TensorDatasetSchema, Schema

from hympi_ml.data.fulldays.loading import DKey, DPath, FullDaysLoader
from hympi_ml.utils import mlflow_log, keras_utils


class FullDaysDataset(Dataset):
    """
    An MLFlow Dataset that represents features and a target derived from fulldays data.

    Fully functional for logging with MLflow, complete with schemas and a detailed profile
    of the days used to generate the datasets. Even includes useful utility functions
    for converting to MemmapBatches.
    """

    def __init__(
        self,
        days: list[str],
        data_path: DPath,
        feature_names: list[DKey],
        target_names: list[DKey],
        scale_ranges: dict[DKey, tuple[Any, Any]] | None = None,
        filters: dict[DKey, list[tuple[Any, Any]]] | None = None,
    ):
        self._feature_names = feature_names
        self._target_names = target_names

        self.scale_ranges = scale_ranges
        self.filters = filters

        self._data_path = data_path
        self.set_days(days)

        self._name = self._create_name()
        self._source = CodeDatasetSource(tags=registry.resolve_tags())

    @classmethod
    def from_base(cls, dataset: Dataset):
        """
        Creates a new FullDaysDataset from the profile of a base Dataset (if applicable).

        Note that this will only work if the profile can be parsed as a FullDaysDataset!
        """
        profile = json.loads(dataset.profile)

        days = profile["days"]
        feature_names = list(profile["feature_shapes"].keys())

        if "target_name" in profile:
            target_names = [profile["target_name"]]
        else:
            target_names = list(profile["target_shapes"].keys())

        data_path = profile["data_path"]

        scale_ranges = profile["scale_ranges"]
        filters = profile["filters"]

        return cls(days, DPath[data_path], feature_names, target_names, scale_ranges, filters)

    def set_days(self, days: list[str]):
        """
        Sets the days that this dataset will load from.
        """
        self._days = days
        self._loader = FullDaysLoader(days, self._data_path)
        self._digest = self._compute_digest()

    def _apply_filters(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        if self.filters is None:
            return dataset

        def range_filter(datas):
            passes = True

            for name, ranges in self.filters.items():
                for mins, maxs in ranges:
                    if datas[name] < mins or datas[name] > maxs:
                        passes = False

            return passes

        return dataset.filter(range_filter)

    def _apply_scaling(self, dataset: tf.data.Dataset, features_only: bool = False):
        if self.scale_ranges is None:
            return dataset

        def scale(datas):
            for name, (minimum, maximum) in self.scale_ranges.items():
                minimum = tf.cast(minimum, tf.float64)
                maximum = tf.cast(maximum, tf.float64)

                if name in self._feature_names or (not features_only and name in self._target_names):
                    datas[name] = tf.divide(tf.subtract(datas[name], minimum), tf.subtract(maximum, minimum))

            return datas

        return dataset.map(scale, num_parallel_calls=tf.data.AUTOTUNE)

    def _split_features_targets(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        def split(datas):
            features = {k: v for k, v in datas.items() if k in self._feature_names}
            targets = {k: v for k, v in datas.items() if k in self._target_names}

            return (features, targets)

        return dataset.map(split, num_parallel_calls=tf.data.AUTOTUNE)

    def as_tf_dataset(self, scale_targets: bool = True, keys: list[DKey] | None = None) -> tf.data.Dataset:
        """
        Generates a TensorFlow Dataset for the features and targets in this FullDaysDataset.
        By default, the dataset has elements in the form of a tuple containing (features, targets) as dictionaries.
        If keys are defined, a dataset containing only those keys will be generated (also as a dictionary).

        Args:
            scale_targets (bool, optional): Whether to scale the targets of the dataset when generating.
                Note that this won't apply if the "keys" parameter is set. Defaults to True.
            keys (list[DKey] | None, optional): A set of specific keys that the dataset will contain. Note that
                this will result in just those keys, not features and targets. Defaults to None.

        Returns:
            tf.data.Dataset: A TensorFlow Dataset
        """
        # zip all features
        names = self._feature_names + self._target_names
        if keys is not None:
            names += keys

        data_dict = {str(key): self.loader.get_tf_dataset(key) for key in names}
        ds = tf.data.Dataset.zip(data_dict)

        ds = self._apply_filters(ds)
        ds = self._apply_scaling(ds, features_only=not scale_targets)

        # handle key filtering and splitting into features and targets
        if keys is not None:
            ds = ds.map(lambda d: {k: v for k, v in d.items() if k in keys})
        else:
            ds = self._split_features_targets(ds)

        return ds

    @property
    def element_spec(self):
        return self.as_tf_dataset().element_spec

    @property
    def feature_specs(self) -> dict[str, tf.TensorSpec]:
        return self.element_spec[0]

    @property
    def feature_shapes(self):
        return {name: tuple(spec.shape.as_list()) for (name, spec) in self.feature_specs.items()}

    @property
    def target_specs(self) -> dict[str, tf.TensorSpec]:
        return self.element_spec[1]

    @property
    def target_shapes(self):
        return {name: tuple(spec.shape.as_list()) for (name, spec) in self.target_specs.items()}

    def get_targets(self, scaling: bool = True) -> dict[str, np.ndarray]:
        """
        Returns a dictionary set of the target data as numpy arrays.
        Useful for working with the entire set of truth data.

        Note: This uses a Keras model to extract the data faster than simply converting the
        tensorflow dataset to a ndarray. This results in using any selected GPUs which will use its
        memory. Keep this in mind when running this for a large set of training data.

        Args:
            scaling (bool, optional): Whether to scale the data according to the scale ranges. Defaults to True.

        Returns:
            dict[str, np.ndarray]: The dictionary containing the numpy arrays of target data.
        """
        identity_model = keras_utils.create_identity_model(self.target_shapes)
        # create the targets-only dataset (ensure quick loading with prefetch and parallel calls)
        targets_dataset = (
            self.as_tf_dataset(scale_targets=scaling)
            .map(lambda _, y: y, num_parallel_calls=tf.data.AUTOTUNE)  # take only the targets
            .batch(1024, num_parallel_calls=tf.data.AUTOTUNE)  # batch for model input
            .prefetch(tf.data.AUTOTUNE)  # prefetch for improved performance
        )

        pred_dict = keras_utils.predict_dict(identity_model, targets_dataset)

        return {self._target_names[i]: v for i, v in enumerate(pred_dict.values())}

    @property
    def days(self) -> list[str]:
        return self._days

    @property
    def loader(self) -> FullDaysLoader:
        return self._loader

    @property
    def source(self) -> DatasetSource:
        return self._source

    @property
    def profile(self) -> dict[str, Any]:
        return {
            "days": self.days,
            "data_path": self._data_path.name,
            "feature_shapes": self.feature_shapes,
            "target_shapes": self.target_shapes,
            "scale_ranges": self.scale_ranges,
            "filters": self.filters,
        }

    @property
    def schema(self) -> TensorDatasetSchema:
        features_schema = Schema(
            [
                mlflow.types.TensorSpec(np.dtype(spec.dtype.as_numpy_dtype), spec.shape.as_list(), name)
                for (name, spec) in self.feature_specs.items()
            ]
        )

        targets_schema = Schema(
            [
                mlflow.types.TensorSpec(np.dtype(spec.dtype.as_numpy_dtype), spec.shape.as_list(), name)
                for (name, spec) in self.target_specs.items()
            ]
        )

        return TensorDatasetSchema(features=features_schema, targets=targets_schema)

    def _create_name(self) -> str:
        """
        Generates a name for this dataset based on features if none
        is provided during initialization.
        """
        feature_names = "+".join(list(self._feature_names))
        return feature_names

    def _compute_digest(self) -> str:
        """
        Computes a digest for this dataset if none is proivded during initialization.
        """
        final_str = f"{self.profile}"

        hash_obj = hashlib.sha256(final_str.encode())
        return hash_obj.hexdigest()[0:8]

    def to_dict(self) -> dict[str, str]:
        """
        Create config dictionary for the dataset.

        Returns a string dictionary containing the following fields: name, digest, source, source
        type, schema, and profile.
        """

        config = super().to_dict()
        config.update(
            {
                "schema": json.dumps(self.schema.to_dict()),
                "profile": json.dumps(self.profile),
            }
        )

        return config

    def log(self, context: str):
        """
        Logs this dataset for the current MLFlow run with a provided "context" label
        """
        mlflow.log_input(self, context)

    def get_input_layers(self) -> dict[str, Any]:
        """
        Returns a dictionary of input layers for building Keras models based on the list
        features in this dataset.
        """
        return {name: keras.Input(shape, name=name) for name, shape in self.feature_shapes.items()}

    def get_output_layers(self, activation="linear") -> dict[str, keras.layers.Dense]:
        """
        Returns a dictionary of input layers for building Keras models based on the list
        features in this dataset.
        """
        output_layers = {}

        for name, shape in self.target_shapes.items():
            if shape == ():
                units = 1
            else:
                units = shape[0]

            output_layers[name] = keras.layers.Dense(units, activation=activation, name=name)

        return output_layers

    def create_unscale_model(self, model: keras.Model) -> keras.Model:
        """
        Creates a new model that automatically unscales the output (if applicable).
        Useful for metric evaluation so that unscaling is automatic.

        Args:
            model (Model): The model that will be used as part of the unscaling model.
                Note: It must have the same input as the features of this dataset.

        Returns:
            Model: A new model with the same inputs but outputs unscaled targets (if applicable).
        """
        inputs = {i.name : i for i in model.inputs}
        model_out = model(inputs)
        
        def unscale(out, name):
            if self.scale_ranges is not None and name in self.scale_ranges:
                minimum, maximum = self.scale_ranges[name]
                return out * (maximum - minimum) + minimum

            return out

        outputs = {name : keras.layers.Lambda(lambda x: unscale(x, name), name=name)(model_out[name]) for name in self._target_names}

        return keras.Model(inputs, outputs)

    def evaluate(
        self,
        model: keras.Model,
        metrics: list[Any],
        context: str,
        batch_size: int = 1024,
        unscale: bool = False,
        log: bool = False,
    ) -> dict[str, Any]:
        """
        Evalutes using a model with inputs and outputs that match the features and targets of
        this dataset. Optionally unscales the targets to get unscaled evaluations.
        """
        if unscale:
            model = self.create_unscale_model(model)
            model.compile(optimizer=model.optimizer, metrics=metrics, loss=model.loss)
            ds = self.as_tf_dataset(scale_targets=False)
        else:
            ds = self.as_tf_dataset(scale_targets=True)

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        eval_dict = model.evaluate(ds, return_dict=True)
        eval_dict = {f"{context}_{k}": v for k, v in eval_dict.items() if "loss" not in k}

        if log:
            mlflow.log_metrics(eval_dict)

        return eval_dict

    def predict(self, model: keras.Model, batch_size: int = 1024, unscale: bool = False) -> dict[str, np.ndarray]:
        """
        Predicts using a model with inputs and outputs that match the features and targets of
        this dataset.

        Note: This will predict on the ENTIRE dataset. This may take a while and use lots of memory.
        Consider forcing CPU usage if GPU memory runs out of memory.
        """
        ds = self.as_tf_dataset().batch(batch_size).prefetch(tf.data.AUTOTUNE)

        if unscale:
            model = self.create_unscale_model(model)
            
        preds = model.predict(ds)

        for k, v in preds.items():
            if v.shape[1] == 1:
                preds[k] = v.flatten()
        
        return preds


def get_split_datasets(
    data_path: DPath,
    feature_names: list[DKey],
    target_names: list[DKey],
    train_days: list[str],
    validation_days: list[str],
    test_days: list[str],
    scale_ranges: dict[DKey, tuple[Any, Any]] | None = None,
    filters: dict[DKey, list[tuple[Any, Any]]] | None = None,
    autolog: bool = False,
) -> tuple[FullDaysDataset, FullDaysDataset, FullDaysDataset]:
    """
    Gets data from a set of three FullDaysDataset and returns a tuple in the form of (train, validation, test)
    """

    train = FullDaysDataset(train_days, data_path, feature_names, target_names, scale_ranges, filters)
    validation = FullDaysDataset(validation_days, data_path, feature_names, target_names, scale_ranges, filters)
    test = FullDaysDataset(test_days, data_path, feature_names, target_names, scale_ranges, filters)

    if autolog:
        train.log("train")
        test.log("test")
        validation.log("validation")

    return (train, validation, test)


def get_datasets_from_run(run_id: str) -> dict[str, FullDaysDataset]:
    """
    Given a mlflow run, parses the datasets and creates a dictionary with a key
    as the context tag of the dataset and the values as the fully parsed FullDaysDataset
    """
    datasets = mlflow_log.get_datasets_by_context(run_id)
    return {k: FullDaysDataset.from_base(v) for k, v in datasets.items()}
