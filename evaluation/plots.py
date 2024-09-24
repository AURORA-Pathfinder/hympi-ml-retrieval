from typing import List, Optional
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from keras.models import Model

from preprocessing.fulldays import FullDaysDataset
from preprocessing.memmap import MemmapSequence


def plot_pred_error(model: Model,
                    data: FullDaysDataset, 
                    index: Optional[int] = None,
                    x_label: str = "Index",
                    y_label: str = "Value",
                    show: bool = True) -> Figure:
    '''
    Returns a figure containing two plots: the bottom showing the predicted
    vs. truth graphs at a given index (assuming a profile) and the top showing
    the error between those graphs.

    Note: The index will be selected at random from the input data if no index
    is provided.
    '''

    if index is None:
        rng = np.random.default_rng()
        index = rng.choice(data.count)

    inputs = data.get_inputs(index)

    pred = model(inputs, training=False)[0]
    truth = data.get_truth(index)

    err = pred - truth
    print(f"MAE: {np.average(np.abs(err))}")

    fig, (err_ax, pred_ax) = plt.subplots(2, 1, sharex=True)

    # Draw the error
    color = 'tab:red'
    err_ax.set_ylabel('Error', color=color)
    err_ax.plot(err, color=color, label="Abs. Error", linewidth=1)
    err_ax.tick_params(axis='y', labelcolor=color)

    # Draw pred v. truth
    color = 'tab:blue'
    pred_ax.set_xlabel(x_label)
    pred_ax.set_ylabel(y_label, color=color)
    pred_ax.plot(pred, color='tab:orange', label="Pred", linewidth=1)
    pred_ax.plot(truth, color=color, label="True", linewidth=1.5)
    pred_ax.tick_params(axis='y', labelcolor=color)

    fig.legend(loc="lower left")
    fig.suptitle(f"Pred. vs Truth and Error for Index {index}")

    if show:
        plt.show()

    return fig


def plot_mae_per_level(error: np.ndarray, show: bool) -> Figure:
    '''
    Returns a figure plotting the MAE for each level in a profile. 
    This is useful for understanding which levels may be resulting in the most error.
    '''
    fig = plt.figure()
    levels = error.shape[1]

    plt.plot([np.average(np.abs(error[:,i])) for i in range(levels)])

    if show:
        plt.show()

    return fig

def plot_variance_per_level(preds: np.ndarray, show: bool) -> Figure:
    '''
    Calculates the variance on a set of predictions for a model. Useful for seeing if the model
    is simply reducing to the mean instead of actually.
    '''
    fig = plt.figure()
    levels = preds.shape[1]

    plt.plot([np.var(preds[:, i]) for i in range(levels)])

    if show:
        plt.show()

    return fig