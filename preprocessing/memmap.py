from typing import List, Optional, Tuple
import math
import collections.abc

import numpy as np
import keras.utils

class MemmapSequence(collections.abc.Sequence):
    '''
    A wrapper class for a list of memmaps that represent a single, combined dataset.

    Once initialized, allows for quickly indexing and slicing into the set of memmaps as if it were one concatenated array.
    '''
    def __init__(self, memmaps: List[np.memmap]):
        ref_shape = memmaps[0].shape[1:]
        for m in memmaps:
            if m.shape[1:] != ref_shape:
                raise Exception(f"Not all memmaps have the same shape! Should be {ref_shape}, but got {m.shape[1:]} from {m.filename}.")
            
        self.memmaps = memmaps

    @classmethod
    def from_files(cls, file_paths: List[str]):
        '''
        Creates a new Memmaps from a set of '.npy' file paths by loading each file in "read" mode.
        '''
        mmaps = [np.load(f, mmap_mode='r') for f in file_paths]
        return cls(mmaps)
    
    def __len__(self):
        '''
        Returns the total size by taking the sum of the first shape value in each memmaps.
        '''
        return sum(map(lambda m: m.shape[0], self.memmaps))

    def __getitem__(self, val: int | slice | np.ndarray) -> np.memmap | np.ndarray:
        '''
        Returns the value at a specific index or slice (or set of indices) from the list of memmaps as if it were
        one concatenated list. 
        
        Note that, if the value is an index or slice within a single file, then 
        the output will be a numpy memmap. However, if a slice traverses multiple files, then an ndarray 
        will be returned due to limitations when concatenating numpy memmaps.
        '''

        if isinstance(val, (int, np.int_)):
            index = val
            file_index = 0

            for m in self.memmaps:
                m_size = m.shape[0]

                if index >= 0 and index < m_size:
                    break

                file_index += 1
                index -= m_size;

            return self.memmaps[file_index][index]

        if isinstance(val, slice):
            start = val.start
            stop = val.stop

            slices = []

            for memmap in self.memmaps:
                size = memmap.shape[0]

                if start < size and stop > 0:
                    clamp_stop = min(size, stop)
                    clamp_start = max(start, 0)

                    data = memmap[clamp_start:clamp_stop:val.step]

                    slices.append(data)

                start -= size
                stop -= size
                
            if len(slices) == 1:
                return slices[0]
            else:
                return np.concatenate(slices)
            
        if isinstance(val, np.ndarray):
            samples = []
            for index in val:
                samples.append(self[index])

            return np.array(samples)
    
    @property
    def data_shape(self) -> Tuple:
        '''
        Returns the shape of the first index of the dataset
        '''
        shape = self.memmaps[0][0].shape

        if shape == ():
            return (1,)
        
        return shape

    @property
    def shape(self) -> Tuple:
        '''
        Returns the shape of the combined set of the memmaps
        '''
        return (self.__len__(),) + self.data_shape


class MemmapBatches(keras.utils.Sequence):
    '''
    A Keras Sequence that generates data batches from MemmapSequences.

    Meant to be used as both the features and targets for training models with Keras.
    '''
    def __init__(self, 
                 features: List[MemmapSequence], 
                 target: MemmapSequence, 
                 batch_size: int, 
                 shuffle: bool = True,
                 shuffle_seed: Optional[int] = None):
        
        self.features = features
        self.target = target

        self.batch_size = batch_size

        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.rng = np.random.default_rng(shuffle_seed)

    def on_epoch_end(self):
        self.rng = np.random.default_rng(self.shuffle_seed)

    def __getitem__(self, index):
        if self.shuffle:
            index = self.rng.choice(self.__len__(), replace=False)

        start = index * self.batch_size
        stop = start + self.batch_size

        x = [feature[start:stop] for feature in self.features]
        y = self.target[start:stop]

        return x, y
    
    def __len__(self):
        return math.ceil(len(self.features[0]) / self.batch_size)