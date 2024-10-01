from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from keras.models import Model

from data.fulldays.dataset import FullDaysDataset
import evaluation.metrics as metrics


def plot_pred_error(
    model: Model,
    data: FullDaysDataset,
    index: Optional[int] = None,
    x_label: str = "Index",
    y_label: str = "Value",
    show: bool = True,
) -> Figure:
    """
    Returns a figure containing two plots: the bottom showing the predicted
    vs. truth graphs at a given index (assuming a profile) and the top showing
    the error between those graphs.

    Note: The index will be selected at random from the input data if no index
    is provided.
    """

    if index is None:
        rng = np.random.default_rng()
        index = rng.choice(data.count)

    inputs = data.get_inputs(index)

    pred = model.predict_step(inputs)[0]
    truth = data.get_truth(index)

    err = pred - truth
    print(f"MAE: {np.average(np.abs(err))}")

    fig, (err_ax, pred_ax) = plt.subplots(2, 1, sharex=True)

    # Draw the error
    color = "tab:red"
    err_ax.set_ylabel("Error", color=color)
    err_ax.plot(err, color=color, label="Abs. Error", linewidth=1)
    err_ax.tick_params(axis="y", labelcolor=color)

    # Draw pred v. truth
    color = "tab:blue"
    pred_ax.set_xlabel(x_label)
    pred_ax.set_ylabel(y_label, color=color)
    pred_ax.plot(pred, color="tab:orange", label="Pred", linewidth=1)
    pred_ax.plot(truth, color=color, label="True", linewidth=1.5)
    pred_ax.tick_params(axis="y", labelcolor=color)

    fig.legend()
    fig.suptitle(f"Pred. vs Truth and Error for Index {index}")

    if show:
        plt.show()

    return fig


def plot_mae_per_level(pred: np.ndarray, truth: np.ndarray, context: str, show: bool) -> Figure:
    """Generates a plot showing mean absolute error (MAE) for each sigma level."""

    mae_per_level = metrics.mae_per_level(pred, truth)

    fig = plt.figure()
    plt.plot(mae_per_level)

    fig.suptitle(f"MAE Per Level ({context})")

    if show:
        plt.show()

    return fig


def plot_variance_per_level(pred: np.ndarray, show: bool) -> Figure:
    """
    Calculates the variance on a set of predictions for a model.
    Useful for seeing if the model is simply reducing to the mean.
    """
    var_per_level = metrics.var_per_level(pred)

    fig = plt.figure()
    plt.plot(var_per_level)

    if show:
        plt.show()

    return fig


def plot_pred_truth_var(pred: np.ndarray, truth: np.ndarray, context: str, show: bool):
    """
    Returns a figure that plots the variance of both the pred array and the truth array.
    Useful for seeing if the predictions are reducing to the mean of truth. If the
    pred variance is flat in comparison to the truth, reduction to the mean may be occuring.

    The context argument is added, in parenthesis, to the title of the figure.
    For example if context="train", then the title would be: "Variance Per Level (train)"
    """
    var_pred = metrics.var_per_level(pred)
    var_truth = metrics.var_per_level(truth)

    fig = plt.figure()
    plt.plot(var_pred, label="Pred")
    plt.plot(var_truth, label="Truth")
    plt.legend()

    fig.suptitle(f"Pred v. True Variance ({context})")

    if show:
        plt.show()

    return fig
