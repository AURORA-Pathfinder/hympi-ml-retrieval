'''
A module for working with transformation / preprocessing model 
layers (normalization, min max scaling, robust scaling, etc).
'''

from typing import Optional

import numpy as np
from keras.layers import Layer, Normalization
import keras.saving
from sklearn.preprocessing import FunctionTransformer
import tensorflow as tf

from preprocessing.memmap import MemmapSequence


def create_norm_layer(data: np.ndarray | MemmapSequence, random_samples: Optional[int]) -> Normalization:
    '''
    Creates a keras normalization layer adapted from the given data sequence. Define a number of random 
    samples to pull a random subset of the input data.  
    '''
    
    if random_samples is not None:
        rng = np.random.default_rng()
        indices = np.random.choice(data.shape[0], random_samples, replace=False)
        data = data[indices]
    
    mean = data.mean()
    variance = data.var()

    return Normalization(mean=mean, variance=variance)


class Transformer(Layer):
    '''
    A Keras model layer that uses an Scikit Learn FunctionTransformer (any kind of scaler) applied to the input data
    '''
    def __init__(self, transformer: FunctionTransformer):
        '''
        Initializes a new transformer layer by fitting to the input data
        '''
        super().__init__()
        self.transformer = transformer

    def call(self, inputs: tf.Tensor):
        return self.transformer.transform(inputs)
    
    @classmethod
    def from_arr(cls, data: np.ndarray, transformer: FunctionTransformer):
        transformer.fit(data)
        return cls(transformer)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "transformer": keras.saving.serialize_keras_object(self.transformer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        layer_config = config.pop("transformer")
        layer = keras.saving.deserialize_keras_object(layer_config)
        return cls(layer)