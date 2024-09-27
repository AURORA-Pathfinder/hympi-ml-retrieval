"""
A module for working with transformation / preprocessing model
layers (normalization, min max scaling, robust scaling, etc).
"""

from typing import Optional

import numpy as np
from keras.layers import Normalization

from data.memmap import MemmapSequence


def create_norm_layer(
    data: np.ndarray | MemmapSequence, random_samples: Optional[int]
) -> Normalization:
    """
    Creates a keras normalization layer adapted from the given data sequence. Define a number of random
    samples to pull a random subset of the input data.
    """

    if random_samples is not None:
        indices = np.random.choice(data.shape[0], random_samples, replace=False)
        data = data[indices]

    mean = data.mean()
    variance = data.var()

    return Normalization(mean=mean, variance=variance)
