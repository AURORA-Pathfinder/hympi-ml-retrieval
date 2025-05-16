import numpy as np
from keras import backend as k
from keras import losses


def weighted_mae(y_true, y_pred) -> float:
    sigma = 72
    # weights = np.ones(sigma, dtype='float32')
    weights = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
        ],
        dtype="float32",
    )

    # Let's norm it
    # weights /= np.sum(weights)

    # Calculate absolute difference between true and predicted values
    absolute_difference = k.abs(y_true - y_pred)

    # Apply weights to absolute differences
    weighted_absolute_difference = absolute_difference * k.abs(weights)

    # Compute the mean of the weighted absolute differences
    loss = k.mean(weighted_absolute_difference)

    return loss


def pbl_mae(y_true, y_pred) -> float:
    """
    Provided the y_true and y_pred are in the form of 72 sigma levels, returns the mean absolute error
    from levels 58-72 which represent the PBL.

    Args:
        y_true (_type_): The 72 sigma level truth profile data
        y_pred (_type_): The 72 sigma level predicted profile data
    """
    return losses.mae(y_true[-14:], y_pred[-14:])
