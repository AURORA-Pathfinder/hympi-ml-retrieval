"""
A module for working with transformation / preprocessing model
layers (normalization, min max scaling, etc).
"""
import numpy as np
import keras

from hympi_ml.data.memmap import MemmapSequence


def adapt_normalization(data: np.ndarray | MemmapSequence, random_samples: int | None) -> keras.layers.Normalization:
    """
    Creates a keras normalization layer adapted from the given data sequence. Define a number of random
    samples to pull a random subset of the input data.
    """

    if random_samples is not None:
        indices = np.random.choice(data.shape[0], random_samples, replace=False)
        data = data[indices]

    mean = data.mean(axis=0)
    variance = data.var(axis=0)

    return keras.layers.Normalization(mean=mean, variance=variance)


def adapt_minmax(data: np.ndarray | MemmapSequence, random_samples: int | None) -> keras.layers.Normalization:
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

    return create_minmax(mins, maxs)


def create_minmax(min: np.ndarray | float, max: np.ndarray | float) -> keras.layers.Normalization:
    """
    Uses a Keras normalization with specially calculated mean and variance that allows the layer
    to work the same way as a custom minmax layer without resorting to a custom keras layers.

    Args:
        min (np.ndarray): The minimum value or values to create the layer
        max (np.ndarray): The maximum value or values to create the layer

    Returns:
        Normalization: A normalization layer with the correctly calculated mean and variance to simulate a min max scaling layer
    """
    return keras.layers.Normalization(mean=min, variance=np.square(max - min))
