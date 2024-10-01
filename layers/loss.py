import numpy as np
from keras import backend as K


def weighted_mae(y_true, y_pred):
    sigma = 72
    # weights = np.ones(sigma, dtype='float32')
    weights = np.array(
        [
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
        ],
        dtype="float32",
    )

    # Let's norm it
    # weights /= np.sum(weights)

    # Calculate absolute difference between true and predicted values
    absolute_difference = K.abs(y_true - y_pred)

    # Apply weights to absolute differences
    weighted_absolute_difference = absolute_difference * weights

    # Compute the mean of the weighted absolute differences
    loss = K.sum(weighted_absolute_difference) / np.sum(weights)

    return loss
