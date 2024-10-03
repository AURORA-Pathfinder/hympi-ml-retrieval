import numpy as np
import keras
import keras.layers

import tensorflow as tf


class GaussianPerBandLayer(keras.layers.Layer):
    def __init__(self, stds):
        super().__init__()
        self.stds = stds

    def call(self, inputs):
        """
        Takes an input tensor consisting of scaled radiances.
        Returns the radiances with a randomly generated noise applied to it.
        """
        noise = np.array([np.random.normal(scale=x) for x in self.stds], dtype="float32")
        self.band_std = tf.Variable(initial_value=noise, trainable=False)
        return tf.add(inputs, self.band_std)


# TODO: Figure out how noise layers work with reading the noise data

# def stack_noise_layers():
#     inputs = []
#     nd = {}
#     nds = np.load("data/noises.npz")
#     for key in nds.keys():
#         band_noise = np.copy(nds[key])
#         band_noise = np.append(band_noise, 0)
#         nd[key] = band_noise
#         nd[band_noise.shape[0]] = band_noise

#     for key, j in zip(bands, inputs_tmp):
#         inputs.append(GaussianPerBandLayer(nd[key])(j))
