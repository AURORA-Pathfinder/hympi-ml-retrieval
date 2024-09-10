import numpy as np
from keras.layers import Input, Normalization

from preprocessing.memmap import MemmapSequence

def build_input_layer(data: np.ndarray | MemmapSequence, name: str | None = None) -> Input:
    '''
    Creates a keras input layer with a shape based on the given data
    '''
    if isinstance(data, MemmapSequence):
        shape = data.get_shape()[1:]

    elif isinstance(data, np.ndarray):
        shape = data.shape()[1:]

    return Input(shape=shape, name=name)


def create_norm_layer(data: np.ndarray | MemmapSequence, mean: float | None = None, variance: float | None = None) -> Normalization:
    '''
    Creates a keras normalization layer. If mean and variance are not provided, then the layer is adapted from the data.

    WARNING: If the input data represents a massive dataset and mean / variance are not provided, adapting the layer
    will take a while. Use a small chunk of the input data if applicable.
    '''

    norm = Normalization(mean=mean, variance=variance)

    if mean is None or variance is None:
        norm.adapt(data)

    return norm;