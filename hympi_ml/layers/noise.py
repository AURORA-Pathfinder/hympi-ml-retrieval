from typing import List
import math

from keras.layers import Layer
from keras.saving.object_registration import register_keras_serializable

import tensorflow as tf

from hympi_ml.data.fulldays.loading import DKey

ATMS_NEDT = [
    0.7,
    0.8,
    0.9,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.75,
    1.2,
    1.2,
    1.5,
    2.4,
    3.6,
    0.5,
    0.6,
    0.8,
    0.8,
    0.8,
    0.8,
    0.9,
]

HW_BW = [
    0.35,
    0.10,
    0.15,
    0.20,
    0.20,
    0.29,
    0.40,
    0.30,
    1.00,
    0.50,
    0.20,
    3.00,
    2.00,
    1.30,
    3.00,
    3.00,
    3.00,
    3.00,
    3.00,
    3.00,
    3.00,
    3.00,
]


@register_keras_serializable()
class PerBandNoise(Layer):
    def __init__(self, nedt: List[float]):
        super().__init__()
        self.trainable = False
        self.nedt = nedt

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Takes in ATMS data as input and adds the appropriate amount of NEDT.
        Note that the seed for each random value based on the input tensor itself.

        Args:
            inputs (tf.Tensor): The ATMS input data.

        Returns:
            tf.Tensor: The output ATMS tensor with added NEDT.
        """
        # finds seed based on first two values of atms input (each atms input creates deterministic random noise)
        seed = tf.cast(tf.reshape(inputs, [-1])[:2], tf.int32)

        nedt = tf.convert_to_tensor(self.nedt)
        noise = tf.stack([tf.random.stateless_normal(shape=(), stddev=n, seed=seed) for n in tf.unstack(nedt)])

        return tf.add(inputs, noise)


def get_hympi_nedt(bandwidth: float) -> float:
    """Calculates the noise using the algorithm developed by Narges

    Args:
        noise_factor (float): The noise factor.
        tau (float): The tau.
        bandwidth (float): The resolution bandwidth (in GHz)

    Returns:
        float: The noise equivalent differential temperature (NEDT)
    """
    noise_factor = 6.5
    tau = 0.018

    t_sys = 290 * (10 ** (noise_factor / 10) - 1) + 250
    return t_sys / math.sqrt(bandwidth * 1e9 * tau)


@register_keras_serializable()
class HympiNoise(Layer):
    def __init__(self, bandwidth: float):
        super().__init__()
        self.trainable = False
        self.bandwidth = bandwidth

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        flat_inputs = tf.reshape(inputs, [-1])
        # finds seed based on first two values of hympi input (each input creates deterministic random noise)
        seed = tf.cast(flat_inputs[:2], tf.int32)

        nedt = get_hympi_nedt(self.bandwidth)
        noise = tf.random.stateless_normal(shape=inputs.shape[1:], stddev=nedt, seed=seed)

        return tf.add(inputs, noise)


def add_nedt_layer(input_layer: Layer, key: DKey) -> Layer:
    """
    Adds a Noise Equivalent Differential Temperature to the given input layer based on the given DKey.
    If the the DKey does not have an existing nedt, it will simply return the input layer.

    Args:
        input_layer (Layer): The layer to add the noise layer to.
        key (DKey): The DKey to get the correct noise values.

    Returns:
        Layer: A new layer with noise added to the input layer (uses Keras functional layers).
    """
    match key:
        case DKey.ATMS:
            return PerBandNoise(ATMS_NEDT)(input_layer)
        case DKey.HW:
            hw_nedt = [get_hympi_nedt(bw) for bw in HW_BW]
            return PerBandNoise(hw_nedt)(input_layer)
        case DKey.HA | DKey.HB:
            return HympiNoise(0.01)(input_layer)
        case DKey.HC:
            return HympiNoise(0.02)(input_layer)
        case DKey.HD:
            return HympiNoise(0.04)(input_layer)
        case DKey.H1:
            return HympiNoise(0.5)(input_layer)
        case _:
            return input_layer
