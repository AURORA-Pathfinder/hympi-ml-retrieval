from typing import Dict, List, Optional
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from keras.models import Model

from hympi_ml.data.fulldays import FullDaysDataset, units
from hympi_ml.data.fulldays.loading import DKey
import hympi_ml.utils.mlflow_log as mlflow_log


def plot_profiles(profiles: Dict[str, np.ndarray], value_axis: Optional[str] = None, levels_axis: str = "Levels"):
    """
    Given a dictionary of arrays that each represent a single profile, plots them into a common format used for
    profile figures. This includes a vertical, inverted x-axis with proper labels.

    Note: This should be used to replace a matplotlib `plt.plot()` method for any profile.
    This way, all profile plots have a consistent look.

    Args:
        profiles (Dict[str, np.ndarray]):
            A dictionary with keys as the label and values as an ndarray representing a profile
        value_axis (str): The label for the values of the profile (example: "Temperature (K)")
        levels_axis (str): The label for the number of levels in this profile. Defaults to "Levels".
    """
    for label, profile in profiles.items():
        plt.plot(profile, range(len(profile)), ".-", label=label)

    ax = plt.gca()

    if value_axis is None:
        value_axis = "Values"

    ax.set_xlabel(value_axis)
    ax.set_ylabel(levels_axis)
    ax.invert_yaxis()


def plot_mae(model: Model, data: FullDaysDataset, context: str, count: int = 10000, log: bool = False) -> List[Figure]:
    """
    Creates a figure that plots the mean absolute error per level of the
    predicted vs truthful data given a model and a FullDaysDataset.

    Args:
        model (Model): A Keras model to predict from
        data (FullDaysDataset): The dataset used to get truth and prediction data
        context (str): Added to the figure title. Useful for saying which dataset is being used (train, test, etc)
        count (int, optional): The number of data points to predict on. Defaults to 10000.
        show (bool, optional): Whether to show this figure in the console. Defaults to False.
    """
    batches = data.create_batches(count)
    (x, truths) = batches[0]
    preds = model.predict(x, steps=1)

    figs = []

    for i in range(len(truths)):
        pred = preds[i]
        truth = truths[i]

        mae_per_level = np.abs(pred - truth).mean(axis=0)

        fig = plt.figure()
        formatted_unit = units.get_formatted_units(data._target_names[i])
        plot_profiles({"Pred": mae_per_level}, value_axis=formatted_unit)
        plt.legend()

        fig.suptitle(f"MAE Per Level ({count} {context} Values)")

        if log:
            mlflow_log.log_figure(fig)

        figs.append(fig)

    return figs


def plot_var_comp(
    model: Model, data: FullDaysDataset, context: str, count: int = 10000, log: bool = False
) -> List[Figure]:
    """
    Creates a figure that plots the variance of predicted and truth data for comparison.
    This can be useful for seeing if the model may be reducing to the mean of the input dataset.

    Args:
        model (Model): A Keras model to predict from
        data (FullDaysDataset): The dataset used to get truth and prediction data
        context (str): Added to the figure title. Useful for saying which dataset is being used (train, test, etc)
        count (int, optional): The number of data points to predict on. Defaults to 10000.
        show (bool, optional): Whether to show this figure in the console. Defaults to False.
    """
    batches = data.create_batches(count)
    (x, truths) = batches[0]
    preds = model.predict(x, steps=1)

    figs = []

    keys = list(data.targets.keys())

    for i in range(len(truths)):
        pred = preds[i]
        truth = truths[i]

        var_pred = np.var(pred, axis=0)
        var_truth = np.var(truth, axis=0)

        fig = plt.figure()
        value_label = units.get_formatted_units(DKey(keys[i]))
        plot_profiles({"Pred": var_pred, "Truth": var_truth}, value_axis=value_label)
        plt.legend()

        fig.suptitle(f"{keys[i]} Pred v. True Variance ({count} {context} Values)")

        if log:
            mlflow_log.log_figure(fig)

        figs.append(fig)

    return figs


def log_eval_profile_plots(model: Model, train: FullDaysDataset, test: FullDaysDataset, count: int = 10000):
    """
    Logs the useful evaluation profile plots for the train and test sets automatically.

    Currently plots the variance comparison and MAE per level for both the train and test datasets.

    Args:
        model (Model): A Keras model to predict from
        train (FullDaysDataset): The train dataset
        test (FullDaysDataset): The test dataset
        count (int, optional): The number of data points to predict on. Defaults to 10000.
        show (bool, optional): Whether to show this figure in the console. Defaults to False.
    """
    # Test
    plot_var_comp(model=model, data=test, context="Test", count=count, log=True)
    plot_mae(model=model, data=test, context="Test", count=count, log=True)

    # Train
    plot_var_comp(model=model, data=train, context="Train", count=count, log=True)
    plot_mae(model=model, data=train, context="Train", count=count, log=True)
