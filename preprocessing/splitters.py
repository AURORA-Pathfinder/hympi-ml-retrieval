import pickle

import numpy as np
from numpy import ndarray

from abc import ABC, abstractmethod

from preprocessing.DataLoader import DataLoader, DataName

class Splitter(ABC):
    @abstractmethod
    def get_train_test_split(self, dataName: DataName) -> (ndarray, ndarray):
        pass

# Creates the split by loading data and slicing it based on a percentage size for the training set
# also includes options for shuffling the data before slicing
class SizeSplitter(Splitter):
    def __init__(self, data_loader: DataLoader, train_size: float, shuffle: bool) -> None:
        self.data_loader = data_loader
        self.train_size = train_size
        self.shuffle = shuffle

    # TODO: Use Sklearn test_train_split function, it includes an import shuffle seed parameter
    def get_train_test_split(self, dataName: DataName) -> (ndarray, ndarray):
        data = self.data_loader.get_data(dataName)

        num_rows = data.shape[0] # NOTE: This is VERY specific to how our data is organized!
        split_index = int(self.train_size * num_rows)
        indices = np.arange(num_rows)

        if self.shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        return (data[train_indices], data[test_indices])

# Creates the split by loading data and slicing it based on indices from a file path
class FileSplitter(Splitter):
    def __init__(self, data_loader: DataLoader, indices_path: str) -> None:
        self.data_loader = data_loader
        self.indices_path = indices_path
    
    def get_train_test_split(self, dataName: DataName) -> (ndarray, ndarray):
        data = self.data_loader.get_data(dataName)

        with open(self.indices_path, 'rb') as f:
            split = pickle.load(f)
            (train_indices, test_indices) = (split["train_indices"], split["test_indices"])
        
        return (data[train_indices], data[test_indices])


# Creates the split by loading train and test data separately
class LoadSplitter(Splitter):
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_train_test_split(self, dataName: DataName) -> (ndarray, ndarray):
        train = self.train_loader.get_data(dataName)
        test = self.test_loader.get_data(dataName)
        return (train, test)
