from typing import Callable, Dict, List, Optional
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


def plot_profile_eval(
    model: Model,
    data: FullDaysDataset,
    func: Callable[[np.ndarray, np.ndarray], Dict[str, np.ndarray]],
    title: str,
    context: str,
    count: int = 10000,
    log: bool = False,
) -> List[Figure]:
    """
    Plots profile evaluation figures by using a model to predict with some provided data.
    The plot is determined by the 'func' parameter which is a callable that takes a 'pred' and
    'truth' argument and returns a dictionary of profiles to plot.

    Args:
        model (Model): The model to predict with
        data (FullDaysDataset): The data to use for prediction and truth values
        func (Callable[[np.ndarray, np.ndarray], Dict[str, np.ndarray]]): The callable that determines the plot created.
            Input is a 'pred' and 'truth' ndarray and should return a dictionary of ndarray profiles to plot.
        title (str): The main title of the plot
        context (str): The context of the data (train, test, or validation)
        count (int, optional): The number of predictions to make. Defaults to 10000.
        log (bool, optional): Whether to log the plots/figures into MLFlow. Defaults to False.

    Returns:
        List[Figure]: A list of plots, one for each target in the provided dataset
    """
    batches = data.create_batches(count)
    (x, truths) = batches[0]
    preds = model.predict(x, steps=1)

    figs = []

    keys = list(data.targets.keys())

    for i in range(len(truths)):
        pred = preds[i]
        truth = truths[i]

        fig = plt.figure()

        profiles = func(pred, truth)
        value_label = units.get_formatted_units(DKey(keys[i]))
        plot_profiles(profiles, value_axis=value_label)

        plt.legend()

        fig.suptitle(f"{keys[i]} {title} ({count} {context} Values)")

        if log:
            mlflow_log.log_figure(fig)

        figs.append(fig)

    return figs


def get_mae_profile(pred: np.ndarray, truth: np.ndarray) -> Dict[str, np.ndarray]:
    mae_per_level = np.abs(pred - truth).mean(axis=0)

    return {"MAE": mae_per_level}


def get_var_comp_profiles(pred: np.ndarray, truth: np.ndarray) -> Dict[str, np.ndarray]:
    var_pred = np.var(pred, axis=0)
    var_truth = np.var(truth, axis=0)

    return {"Pred": var_pred, "Truth": var_truth}


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
    plot_profile_eval(
        model=model,
        data=test,
        func=get_mae_profile,
        title="MAE Per Level",
        context="Test",
        count=count,
        log=True,
    )

    plot_profile_eval(
        model=model,
        data=test,
        func=get_var_comp_profiles,
        title="Pred v. Truth Variance",
        context="Test",
        count=count,
        log=True,
    )

    # Train
    plot_profile_eval(
        model=model,
        data=train,
        func=get_mae_profile,
        title="MAE Per Level",
        context="Train",
        count=count,
        log=True,
    )

    plot_profile_eval(
        model=model,
        data=train,
        func=get_var_comp_profiles,
        title="Pred v. Truth Variance",
        context="Train",
        count=count,
        log=True,
    )
