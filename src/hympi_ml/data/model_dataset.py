from typing import Mapping, cast
import json
import hashlib

import numpy as np

import torch.utils.data

import mlflow
import mlflow.types
import mlflow.data
import mlflow.data.dataset
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.tracking.context import registry
from mlflow.data.schema import TensorDatasetSchema

from hympi_ml.data import DataSpec, DataSource
from hympi_ml.utils import mlf


class ModelDataset(mlflow.data.dataset.Dataset, torch.utils.data.Dataset):
    """
    Combines sets of DataSpec for features and targets with the same DataSource.

    It is a combination of an MLFlow dataset and a PyTorch dataset (inherits both!).
    As a PyTorch dataset, it is fully capable of being indexed and used in a DataLoader.
    As an MLFlow dataset, it can be logged as part of an MLFlow run and later reproduced for further evaluation, etc.
    """

    def __init__(
        self,
        data_source: DataSource,
        features: Mapping[str, DataSpec],
        targets: Mapping[str, DataSpec],
        batch_size: int,
        name: str | None = None,
    ):
        self.batch_size = batch_size

        self._data_source = data_source

        self.features = features
        self.targets = targets

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

        if "batch_size" in profile.keys():
            batch_size = profile["batch_size"]
        else:
            batch_size = 1024

        return cls(source, features, targets, batch_size)

    def __len__(self):
        return int(self._data_source.sample_count / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size

        features_batch = {}

        for name, feature_spec in self.features.items():
            raw_batch = feature_spec.load_raw_slice(
                source=self._data_source, start=start, end=end
            )
            raw_batch = torch.from_numpy(
                np.array(raw_batch, copy=True, dtype=np.float32)
            )

            if len(raw_batch.shape) == 1:
                raw_batch = raw_batch.reshape(-1, 1)

            features_batch[name] = raw_batch

        targets_batch = {}

        for name, target_spec in self.targets.items():
            raw_batch = target_spec.load_raw_slice(
                source=self._data_source, start=start, end=end
            )
            raw_batch = torch.from_numpy(
                np.array(raw_batch, copy=True, dtype=np.float32)
            )

            if len(raw_batch.shape) == 1:
                raw_batch = raw_batch.reshape(-1, 1)

            targets_batch[name] = raw_batch

        return (features_batch, targets_batch)

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
            "batch_size": self.batch_size,
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
        feature_names = "+".join(list(self.features.keys()))
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


def get_split_datasets(
    features: Mapping[str, DataSpec],
    targets: Mapping[str, DataSpec],
    train_source: DataSource,
    validation_source: DataSource,
    test_source: DataSource,
    batch_size: int = 1024,
    autolog: bool = False,
) -> tuple[ModelDataset, ModelDataset, ModelDataset]:
    """
    Automatically creates three ModelDataset with the same features, targets, and batch size but with
    different DataSource for train, validation, and test.

    Returns:
        A tuple of ModelDataset in the form of (train, validation, test).
    """

    train = ModelDataset(train_source, features, targets, batch_size)
    validation = ModelDataset(validation_source, features, targets, batch_size)
    test = ModelDataset(test_source, features, targets, batch_size)

    if autolog:
        train.log("train")
        test.log("test")
        validation.log("validation")

    return (train, validation, test)


def get_datasets_from_run(run_id: str) -> dict[str, ModelDataset]:
    """
    Given a mlflow run id, parses the datasets and creates a dictionary with a key
    as the context tag of the dataset and the values as the fully parsed ModelDataset.
    """
    datasets = mlf.get_datasets_by_context(run_id)
    return {
        k: ModelDataset.from_base(cast(mlflow.data.dataset.Dataset, v))
        for k, v in datasets.items()
    }
