from typing import List
import copy

from sklearn.preprocessing import FunctionTransformer

from preprocessing.DataArray import DataArray
from preprocessing.ModelIO import ModelIO
from preprocessing.DataLoader import DataName
from preprocessing.splitters import Splitter

class DataPreprocessor:
    def __init__(self, dataName: DataName, transformer: FunctionTransformer, splitter: Splitter) -> None:
        self.dataName = dataName
        self.transformer = transformer
        self.splitter = splitter

    # creates the train and test dataset in the form of (train, test) both of type DataArray
    def create_split_datasets(self) -> (DataArray, DataArray):
        (train_data, test_data) = self.splitter.get_train_test_split(self.dataName)
        
        train = DataArray(train_data, copy.copy(self.transformer))
        test = DataArray(test_data, copy.copy(self.transformer))

        return (train, test)

# Creates the train and test dataset in a tuple (train, test) both of type ModelIO
def create_modelIOs(feature_processors: List[DataPreprocessor], target_processor: DataPreprocessor) -> (ModelIO, ModelIO):
    split_features = [feature_processor.create_split_datasets() for feature_processor in feature_processors]
    
    train_features = [split_feature[0] for split_feature in split_features]
    test_features = [split_feature[1] for split_feature in split_features]

    (train_target, test_target) = target_processor.create_split_datasets()

    train = ModelIO(features=train_features, target=train_target)
    test = ModelIO(features=test_features, target=test_target)

    return (train, test)