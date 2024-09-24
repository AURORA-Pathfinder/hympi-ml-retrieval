from typing import List, Optional, Sequence, Tuple, Dict, Any
from enum import Enum, auto
import json
import hashlib

import numpy as np
import mlflow
from mlflow.data.dataset import Dataset, DatasetSource
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.tracking.context import registry
from mlflow.data.schema import TensorDatasetSchema, Schema
from mlflow.types import TensorSpec

from keras.layers import Input
from keras.models import Model
from preprocessing.memmap import MemmapBatches, MemmapSequence

class DataName(Enum):
    '''
    An Enum that contains the list of datasets that could be loaded.
    '''
    hsel = auto()
    hsel_new = auto()
    ha = auto()
    hb = auto()
    hc = auto()
    hd = auto()
    hw = auto()
    hmw = auto()

    atms = auto()

    labels_scalar = auto()
    latitude = auto()
    longitude = auto()
    pblh = auto()
    land_fraction = auto()
    surface_pressure = auto()
    surface_temperature = auto()

    labels_table = auto()
    pressure = auto()
    temperature = auto()
    water_vapor = auto()


class FullDaysLoader:
    '''
    A utility class for loading Fulldays data initialized with a given number of days in the YYYYMMDD format.
    '''
    def __init__(self, days: List[str]) -> None:
        self.days = days

    def get_data(self, dataName: DataName | str) -> MemmapSequence:
        '''
        Returns a MemmapSequence from a given DataName on the days initialized in this loader.
        '''
        # convert dataName from a string to enum (if applicable)
        if isinstance(dataName, str):
            dataName = DataName[dataName]

        memmaps = [self.find_memmap(day, dataName) for day in self.days]
        return MemmapSequence(memmaps)

    
    def find_memmap(self, day: str, dataName: DataName) -> np.memmap:
        '''
        Matches a given DataName to a specific memmap in a file on disk.
        '''
        dir_path = f"/data/nature_run/fulldays_reduced_evenmore/{day}/"

        hsel = np.load(dir_path + "hsel.npy", mmap_mode='r')
        hsel_new = np.load(dir_path + "hsel_new.npy", mmap_mode='r')
        hw = np.load(dir_path + "hw.npy", mmap_mode='r')
        atms = np.load(dir_path + "mh.npy", mmap_mode='r')
        scalar = np.load(dir_path + "scalar.npy", mmap_mode='r')
        table = np.load(dir_path + "table.npy", mmap_mode='r')

        match dataName:
            case DataName.hsel: return hsel
            case DataName.hsel_new: return hsel_new
            case DataName.ha: return hsel[:, 0:471]
            case DataName.hb: return hsel[:, 471:932]
            case DataName.hc: return hsel[:, 932:1433]
            case DataName.hd: return hsel[:, 1433:1934]
            case DataName.hmw: return hsel[:, -1]
            case DataName.hw: return hw

            case DataName.atms: return atms

            case DataName.labels_scalar: return scalar
            case DataName.latitude: return scalar[:, 0]
            case DataName.longitude: return scalar[:, 1]
            case DataName.land_fraction: return scalar[:, 2]
            case DataName.surface_pressure: return scalar[:, 3]
            case DataName.surface_temperature: return scalar[:, 4]
            case DataName.pblh: return scalar[:, 13]

            case DataName.labels_table: return table
            case DataName.pressure: return table[:, :, 0]
            case DataName.temperature: return table[:, :, 1]
            case DataName.water_vapor: return table[:, :, 2]
            
            case _: raise Exception(f"No match for {str(dataName)} found in {dir_path}")


def get_split_data(dataName: DataName | str, 
                   train_loader: FullDaysLoader, 
                   val_loader: FullDaysLoader,
                   test_loader: FullDaysLoader
                   ) -> Tuple[MemmapSequence, MemmapSequence, MemmapSequence]:
    '''
    Gets data from a set of three FullDaysLoader and returns a tuple in the form of (train, validation, test)
    '''
    
    train = train_loader.get_data(dataName)
    validation = val_loader.get_data(dataName)
    test = test_loader.get_data(dataName)

    return (train, validation, test)


class FullDaysDataset(Dataset):
    '''
    An MLFlow Dataset that represents features and a target derived from fulldays data.
    
    Fully functional for logging with MLflow, complete with schemas and a detailed profile
    of the days used to generate the datasets. Even includes useful utility functions
    for converting to MemmapBatches.
    '''
    def __init__(
        self,
        days: List[str],
        feature_names: List[DataName],
        target_name: DataName,
        source: Optional[DatasetSource] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        self._days = days

        self._feature_names = feature_names
        self._target_name = target_name

        self._loader = FullDaysLoader(days)
        
        self._features: Dict[str, MemmapSequence] = {}
        for data_name in feature_names:
            seq = self._loader.get_data(data_name)
            self._features.update({data_name.name: seq})

        self._target = self._loader.get_data(target_name)

        if source is None:
             source = CodeDatasetSource(tags=registry.resolve_tags())

        if name is None:
            name = self._create_name()

        self._source = source

        super().__init__(source=source, name=name, digest=digest)

    @property
    def days(self) -> List[str]:
        return self._days

    @property
    def count(self) -> int:
        return len(self.features.keys[0])
    
    @property
    def features(self) -> Dict[str, MemmapSequence]:
        return self._features
    
    @property
    def feature_shapes(self) -> Dict[str, Tuple]:
        shapes: Dict[DataName, Tuple] = {}

        for name in self.features.keys():
            shapes.update({name: self.features[name].data_shape})

        return shapes
    
    @property
    def target(self) -> MemmapSequence:
        return self._target
    
    @property
    def target_name(self) -> str:
        return self._target_name.name
    
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
    def schema(self) -> Schema:
        features_schema = Schema([
            TensorSpec(seq[0].dtype, seq.data_shape, name) 
            for (name, seq) in self.features.items()]
        )

        target_schema = Schema([
            TensorSpec(self.target[0].dtype, self.target.data_shape, self.target_name) 
        ])

        return TensorDatasetSchema(features=features_schema, targets=target_schema)

    @property
    def count(self) -> int:
        '''
        Returns the size of the dataset (simply the number of data points)
        ''' 
        return len(list(self.features.values())[0])
    
    def _create_name(self) -> str:
        '''
        Generates a name for this dataset based on features and target if none
        is provided during initialization.
        '''
        feature_names = '+'.join(list(self.features.keys()))
        target_name = f"={self.target_name}"

        return feature_names+target_name

    def _compute_digest(self) -> str:
        '''
        Computes a digest for this dataset if none is proivded during initialization.
        '''
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
        '''
        Logs this dataset for the current MLFlow run with a provided "context" label
        '''
        mlflow.log_input(self, context)
    
    def create_batches(self, batch_size: int) -> MemmapBatches:
        return MemmapBatches(list(self.features.values()), self.target, batch_size)

    def get_latlon(self) -> Tuple[MemmapSequence, MemmapSequence]:
        lat = self.loader.get_data(DataName.latitude)
        lon = self.loader.get_data(DataName.longitude)
        return lat, lon
    
    def get_inputs(self, index: int) -> Dict[str, Any]:
        '''
        Returns a dictionary that represents the value at a specific index of the feature datasets.

        This can be used to get a specific input to predict on a model
        '''
        return {k: v[index] for k, v in self.features.items()}
    
    def get_truth(self, index: int) -> Any:
        '''
        Returns the value at a specific index of the target dataset.

        This can be used to get the truth value at a defined index for evaluations.
        '''
        return self.target[index]
    
    def get_input_layers(self) -> Dict[str, Input]:
        '''
        Returns a dictionary of input layers for building Keras models based on the list
        features in this dataset.
        '''
        return {k: Input(v.data_shape, name=k) for k, v in self.features.items()}
    
    def predict(self, model: Model, batch_size: int = 1024) -> np.ndarray:
        '''
        Predicts using a model with inputs and outputs that match the features and targets of
        this dataset.

        Note: This will predict on the ENTIRE dataset. This may take a while and use lots of memory. 
        Consider forcing CPU usage if GPU memory runs out of memory.
        '''
        batches=self.create_batches(batch_size)
        batches.shuffle = False; # disables shuffling to ensure output is in same order as dataset
        return model.predict(batches)