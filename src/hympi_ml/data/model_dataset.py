from typing import Mapping, Any, cast
import json
import hashlib
import copy

import numpy as np

import torch.utils.data

import mlflow
import mlflow.types
import mlflow.data
import mlflow.data.dataset
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.tracking.context import registry
from mlflow.data.schema import TensorDatasetSchema

import keras

from hympi_ml.data import DataSpec, DataSource
from hympi_ml.utils import mlflow_log


class ModelDataset(mlflow.data.dataset.Dataset, torch.utils.data.Dataset):
    """
    An MLFlow Dataset that represents features and a target derived from some loader of data.

    Fully functional for logging with MLflow, complete with schemas and a detailed profile
    of the days used to generate the datasets.
    """

    def __init__(
        self,
        data_source: DataSource,
        features: Mapping[str, DataSpec],
        targets: Mapping[str, DataSpec],
        batch_size: int = 1024,
        name: str | None = None,
    ):
        self.batch_size = batch_size

        self._data_source = data_source

        self.features = features
        self.feature_names = list(features.keys())

        self.targets = targets
        self.target_names = list(targets.keys())

        self._digest = self._compute_digest()

        self._name = name or self._create_name()
        self._source = CodeDatasetSource(tags=registry.resolve_tags())

    @classmethod
    def from_base(cls, dataset: mlflow.data.dataset.Dataset):
        """
        Creates a new ModelDataset from the profile of a base Dataset (if applicable).

        Note that this will only work if the profile can be parsed as a ModelDataset!
        """
        profile = json.loads(dataset.profile)

        source = DataSource.from_dump(json.loads(profile["data_source"]))

        features = {
            k: DataSpec.from_dump(dump)
            for k, dump in json.loads(profile["features"]).items()
        }
        targets = {
            k: DataSpec.from_dump(dump)
            for k, dump in json.loads(profile["targets"]).items()
        }

        return cls(source, features, targets)

    def __len__(self):
        return int(self._data_source.sample_count / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size

        # for filtering we could just see what got removed and reduce that from the sample count??

        features_batch = {}

        for name, feature_spec in self.features.items():
            raw_batch = feature_spec.load_raw_slice(
                source=self._data_source, start=start, end=end
            )
            raw_batch = torch.from_numpy(
                np.array(raw_batch, copy=True, dtype=np.float32)
            )
            # batch = self.features[name].apply_batch(raw_batch)
            features_batch[name] = raw_batch

        targets_batch = {}

        for name, target_spec in self.targets.items():
            raw_batch = target_spec.load_raw_slice(
                source=self._data_source, start=start, end=end
            )
            raw_batch = torch.from_numpy(
                np.array(raw_batch, copy=True, dtype=np.float32)
            )
            # batch = self.targets[name].apply_batch(raw_batch)
            targets_batch[name] = raw_batch

        return (features_batch, targets_batch)

    def get_input_layers(self):
        return {
            k: keras.layers.Input(shape=v.shape, name=k)
            for k, v in self.features.items()
        }

    def get_output_layers(self):
        return {
            k: keras.layers.Dense(v.shape[0], name=k) for k, v in self.targets.items()
        }

    @property
    def feature_shapes(self):
        return {k: v.shape for k, v in self.features.items()}

    @property
    def target_shapes(self):
        return {k: v.shape for k, v in self.targets.items()}

    @property
    def data_source(self) -> DataSource:
        return self._data_source

    @property
    def profile(self) -> str:
        source_dump = self.data_source.model_dump()
        source_dump = json.dumps(source_dump)

        feature_dump = {
            name: feature.model_dump(exclude_none=True)
            for name, feature in self.features.items()
        }
        feature_dump = json.dumps(feature_dump)

        target_dump = {
            name: target.model_dump(exclude_none=True)
            for name, target in self.targets.items()
        }
        target_dump = json.dumps(target_dump)

        prof_dict = {
            "data_source": source_dump,
            "features": feature_dump,
            "targets": target_dump,
        }

        return json.dumps(prof_dict)

    @property
    def schema(self) -> TensorDatasetSchema:
        features_schema = mlflow.types.Schema(
            [
                mlflow.types.TensorSpec(np.dtype(np.float64), list(shape), name)
                for (name, shape) in self.feature_shapes.items()
            ]
        )

        targets_schema = mlflow.types.Schema(
            [
                mlflow.types.TensorSpec(np.dtype(np.float64), list(shape), name)
                for (name, shape) in self.target_shapes.items()
            ]
        )

        return TensorDatasetSchema(features=features_schema, targets=targets_schema)

    def _create_name(self) -> str:
        """
        Generates a name for this dataset based on features if none
        is provided during initialization.
        """
        feature_names = "+".join(list(self.feature_names))
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
                "profile": self.profile,
            }
        )

        return config

    def log(self, context: str):
        """
        Logs this dataset for the current MLFlow run with a provided "context" label
        """
        mlflow.log_input(self, context)

    def get_unscaled_copy(self):
        unscaled_targets = {}

        for name, target in self.targets.items():
            new_target = copy.deepcopy(target)
            new_target.scale_range = None
            unscaled_targets[name] = new_target

        return ModelDataset(
            self.data_source, self.features, unscaled_targets, self.batch_size
        )

    def create_unscale_model(self, model: keras.Model) -> keras.Model:
        """
        Creates a new model that automatically unscales the output (if applicable).
        Useful for metric evaluation so that unscaling is automatic.

        Args:
            model (keras.Model): The model that will be used as part of the unscaling model.
                Note: It must have the same input as the features of this dataset.

        Returns:
            keras.Model: A new model with the same inputs but outputs unscaled targets (if applicable).
        """
        inputs = {i.name: i for i in model.inputs}
        model_out = model(inputs)

        def unscale(out, scale_range):
            if scale_range is not None:
                minimum, maximum = scale_range
                return out * (maximum - minimum) + minimum

            return out

        outputs = {
            name: keras.layers.Lambda(
                unscale,
                name=name,
                output_shape=self.target_shapes[name],
                arguments={"scale_range": target.scale_range},
            )(model_out[name])
            for name, target in self.targets.items()
        }

        return keras.Model(inputs, outputs)

    def evaluate(
        self,
        model: keras.Model,
        metrics: dict[str, list[Any]],
        context: str,
        unscale: bool = False,
        log: bool = False,
    ) -> dict[str, Any]:
        """
        Evalutes using a model with inputs and outputs that match the features and targets of
        this dataset. Optionally unscales the targets to get unscaled evaluations.
        """
        if unscale:
            model = self.create_unscale_model(model)
            ds = self.get_unscaled_copy()
        else:
            ds = self

        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, shuffle=False, num_workers=10, pin_memory=True
        )

        model.compile(metrics=metrics, loss="mae")
        eval_dict = model.evaluate(loader, return_dict=True)
        eval_dict = {
            f"{context}_{k}": v for k, v in eval_dict.items() if "loss" not in k
        }

        if log:
            for k, v in eval_dict.items():
                if "loss" in k:
                    continue

                if isinstance(v, float):
                    mlflow.log_metric(key=k, value=v)
                else:
                    np.save(f"/tmp/{k}.npy", np.array(v.cpu()))
                    mlflow.log_artifact(
                        local_path=f"/tmp/{k}.npy",
                        artifact_path=f"{context}_metrics",
                    )

        return eval_dict


def get_split_datasets(
    features: Mapping[str, DataSpec],
    targets: Mapping[str, DataSpec],
    train_source: DataSource,
    validation_source: DataSource,
    test_source: DataSource,
    autolog: bool = False,
) -> tuple[ModelDataset, ModelDataset, ModelDataset]:
    """
    Gets data from a set of three ModelDataset and returns a tuple in the form of (train, validation, test)
    """

    train = ModelDataset(train_source, features, targets)
    validation = ModelDataset(validation_source, features, targets)
    test = ModelDataset(test_source, features, targets)

    if autolog:
        train.log("train")
        test.log("test")
        validation.log("validation")

    return (train, validation, test)


def get_datasets_from_run(run_id: str) -> dict[str, ModelDataset]:
    """
    Given a mlflow run, parses the datasets and creates a dictionary with a key
    as the context tag of the dataset and the values as the fully parsed FullDaysDataset
    """
    datasets = mlflow_log.get_datasets_by_context(run_id)
    return {
        k: ModelDataset.from_base(cast(mlflow.data.dataset.Dataset, v))
        for k, v in datasets.items()
    }
