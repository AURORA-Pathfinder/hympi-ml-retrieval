from typing import List, Optional, Tuple, Dict, Any
import json
import hashlib

import mlflow.entities
import numpy as np
import mlflow
from mlflow.data.dataset import Dataset, DatasetSource
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.tracking.context import registry
from mlflow.data.schema import TensorDatasetSchema, Schema
from mlflow.types import TensorSpec

from keras.layers import Input
from keras.models import Model

from data.memmap import MemmapBatches, MemmapSequence
from data.fulldays.loading import DKey, FullDaysLoader
from utils import mlflow_logging


class FullDaysDataset(Dataset):
    """
    An MLFlow Dataset that represents features and a target derived from fulldays data.

    Fully functional for logging with MLflow, complete with schemas and a detailed profile
    of the days used to generate the datasets. Even includes useful utility functions
    for converting to MemmapBatches.
    """

    def __init__(
        self,
        days: List[str],
        feature_names: List[DKey],
        target_name: DKey,
        source: Optional[DatasetSource] = None,
        name: Optional[str] = None,
    ):
        self._feature_names = feature_names
        self._target_name = target_name

        self.set_days(days)

        self._name = name or self._create_name()

        self._source = source or CodeDatasetSource(tags=registry.resolve_tags())

    @classmethod
    def from_base(cls, dataset: Dataset):
        """
        Creates a new FullDaysDataset from the profile of a base Dataset (if applicable).

        Note that this will only work if the profile can be parsed as a FullDaysDataset!
        """
        profile = json.loads(dataset.profile)

        days = profile["days"]
        feature_names = list(profile["feature_shapes"].keys())
        target_name = profile["target_name"]

        return cls(days, feature_names, target_name)

    def set_days(self, days: List[str]):
        """
        Sets the days that this dataset will load from.
        """
        self._days = days

        self._loader = FullDaysLoader(days)

        self._features: Dict[str, MemmapSequence] = {}
        for data_name in self._feature_names:
            seq = self._loader.get_data(data_name)
            self._features.update({data_name: seq})

        self._target = self._loader.get_data(self._target_name)

        self._digest = self._compute_digest()

    @property
    def days(self) -> List[str]:
        return self._days

    @property
    def features(self) -> Dict[str, MemmapSequence]:
        return self._features

    @property
    def feature_shapes(self) -> Dict[str, Tuple]:
        return {k: v.data_shape for k, v in self.features.items()}

    @property
    def target(self) -> MemmapSequence:
        return self._target

    @property
    def target_name(self) -> str:
        return self._target_name

    @property
    def target_shape(self) -> Tuple:
        return self.target.data_shape

    @property
    def loader(self) -> FullDaysLoader:
        return self._loader

    @property
    def source(self) -> DatasetSource:
        return self._source

    @property
    def profile(self) -> Dict[str, Any]:
        return {
            "days": self.days,
            "count": self.count,
            "feature_shapes": self.feature_shapes,
            "target_name": self.target_name,
            "target_shape": self.target_shape,
        }

    @property
    def schema(self) -> TensorDatasetSchema:
        features_schema = Schema(
            [
                TensorSpec(seq[0].dtype, seq.data_shape, name)
                for (name, seq) in self.features.items()
            ]
        )

        target_schema = Schema(
            [TensorSpec(self.target[0].dtype, self.target.data_shape, self.target_name)]
        )

        return TensorDatasetSchema(features=features_schema, targets=target_schema)

    @property
    def count(self) -> int:
        """
        Returns the size of the dataset (simply the number of data points)
        """
        return len(self.target)

    def _create_name(self) -> str:
        """
        Generates a name for this dataset based on features and target if none
        is provided during initialization.
        """
        feature_names = "+".join(list(self.features.keys()))
        target_name = f"={self.target_name}"

        return feature_names

    def _compute_digest(self) -> str:
        """
        Computes a digest for this dataset if none is proivded during initialization.
        """
        feature_list = list(self.feature_shapes.values())
        target_list = list({self.target_name: self.target_shape}.values())

        final_str = f"{feature_list}{target_list}{self.count}{self._days}"

        hash_obj = hashlib.sha256(final_str.encode())
        return hash_obj.hexdigest()[0:8]

    def to_dict(self) -> Dict[str, str]:
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

    def create_batches(self, batch_size: int) -> MemmapBatches:
        return MemmapBatches(list(self.features.values()), self.target, batch_size)

    def get_latlon(self) -> Tuple[MemmapSequence, MemmapSequence]:
        lat = self.loader.get_data(DKey.LATITUDE)
        lon = self.loader.get_data(DKey.LONGITUDE)
        return lat, lon

    def get_inputs(self, index: int) -> Dict[str, Any]:
        """
        Returns a dictionary that represents the value at a specific index of the feature datasets.

        This can be used to get a specific input to predict on a model
        """
        return {k: v[index].reshape(1, -1) for k, v in self.features.items()}

    def get_truth(self, index: int) -> Any:
        """
        Returns the value at a specific index of the target dataset.

        This can be used to get the truth value at a defined index for evaluations.
        """
        return self.target[index]

    def get_input_layers(self) -> Dict[str, Any]:
        """
        Returns a dictionary of input layers for building Keras models based on the list
        features in this dataset.
        """
        return {k: Input(v.data_shape, name=k.name) for k, v in self.features.items()}

    def predict(self, model: Model, batch_size: int = 1024) -> np.ndarray:
        """
        Predicts using a model with inputs and outputs that match the features and targets of
        this dataset.

        Note: This will predict on the ENTIRE dataset. This may take a while and use lots of memory.
        Consider forcing CPU usage if GPU memory runs out of memory.
        """
        batches = self.create_batches(batch_size)

        # disables shuffling to ensure output is in same order as dataset
        batches.shuffle = False

        return model.predict(batches)


def get_split_datasets(
    feature_names: List[DKey],
    target_name: DKey,
    train_days: List[str],
    validation_days: List[str],
    test_days: List[str],
    logging: bool,
) -> Tuple[FullDaysDataset, FullDaysDataset, FullDaysDataset]:
    """
    Gets data from a set of three FullDaysLoader and returns a tuple in the form of (train, validation, test)
    """

    train = FullDaysDataset(
        days=train_days,
        feature_names=feature_names,
        target_name=target_name,
    )

    validation = FullDaysDataset(
        days=validation_days, feature_names=feature_names, target_name=target_name
    )

    test = FullDaysDataset(
        days=test_days, feature_names=feature_names, target_name=target_name
    )

    if logging:
        train.log("train")
        validation.log("validation")
        test.log("test")

    return (train, validation, test)


def get_datasets_from_run(run_id: str) -> Dict[str, FullDaysDataset]:
    """
    Given a mlflow run, parses the datasets and creates a dictionary with a key
    as the context tag of the dataset and the values as the fully parsed FullDaysDataset
    """
    datasets = mlflow_logging.get_datasets_by_context(run_id)
    return {k: FullDaysDataset.from_base(v) for k, v in datasets.items()}
