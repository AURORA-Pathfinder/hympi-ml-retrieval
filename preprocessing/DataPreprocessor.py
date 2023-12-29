from typing import List
import copy

from sklearn.preprocessing import FunctionTransformer

from preprocessing.DataArray import DataArray
from preprocessing.ModelIO import ModelIO
from preprocessing.DataLoader import DataName
from preprocessing.splitters import Splitter

from numpy import ndarray


class DataPreprocessor:
    def __init__(self,
                 dataName: DataName,
                 transformer: FunctionTransformer,
                 splitter: Splitter) -> None:

        self.dataName = dataName
        self.transformer = transformer
        self.splitter = splitter

    # creates the train and test dataset in the form of (train, test)
    #  both of type DataArray
    def create_split_datasets(self) -> (DataArray, DataArray, ndarray, ndarray):
        (train_data, test_data, train_latlon, test_latlon) = self.splitter.get_train_test_split(self.dataName)

        train = DataArray(train_data, copy.copy(self.transformer))
        test = DataArray(test_data, copy.copy(self.transformer))

        return (train, test, train_latlon, test_latlon)


# Creates the train and test dataset in a tuple
#  (train, test) both of type ModelIO
def create_modelIOs(feature_processors: List[DataPreprocessor],
                    target_processor: DataPreprocessor,
                    verbose: bool = False) -> (ModelIO, ModelIO):

    if verbose:
        print("Loading Features...")

    split_features = [feature_processor.create_split_datasets()
                      for feature_processor in feature_processors]

    train_features = [split_feature[0] for split_feature in split_features]
    test_features = [split_feature[1] for split_feature in split_features]

    train_latlon = split_features[0][2]
    test_latlon = split_features[0][3]

    if verbose:
        print("Loading Targets...")

    (train_target, test_target, _, _) = target_processor.create_split_datasets()

    train = ModelIO(features=train_features, target=train_target, latlon=train_latlon)
    test = ModelIO(features=test_features, target=test_target, latlon=test_latlon)

    return (train, test)
