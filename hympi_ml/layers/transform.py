"""
A module for working with transformation / preprocessing model
layers (normalization, min max scaling, etc).
"""

from typing import Optional

import numpy as np
from keras.layers import Normalization

from hympi_ml.data.memmap import MemmapSequence


def create_norm_layer(data: np.ndarray | MemmapSequence, random_samples: Optional[int]) -> Normalization:
    """
    Creates a keras normalization layer adapted from the given data sequence. Define a number of random
    samples to pull a random subset of the input data.
    """

    if random_samples is not None:
        indices = np.random.choice(data.shape[0], random_samples, replace=False)
        data = data[indices]

    mean = data.mean(axis=0)
    variance = data.var(axis=0)

    return Normalization(mean=mean, variance=variance)


def create_minmax_layer(data: np.ndarray | MemmapSequence, random_samples: Optional[int]) -> Normalization:
    """
    Creates a keras normalization layer adapted from the given data sequence. Define a number of random
    samples to pull a random subset of the input data if the data is too large.

    This layer is a minmax layer because it manipulates how the normalization layer is calculated to produce
    the exact same calculations as a minmax layer without resorting to a custom keras layer.
    """

    if random_samples is not None:
        indices = np.random.choice(data.shape[0], random_samples, replace=False)
        data = data[indices]

    mins = data.min(axis=0)
    maxs = data.max(axis=0)

    return Normalization(mean=mins, variance=np.square(maxs - mins))
