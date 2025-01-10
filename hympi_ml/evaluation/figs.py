from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from keras.models import Model

from hympi_ml.data.fulldays import units, DKey
from hympi_ml.data.fulldays.dataset import FullDaysDataset, get_datasets_from_run
import hympi_ml.utils.mlflow_log as mlflow_log


class EvalFigure(ABC):
    """The base class that contains functions for generating a specfic kind of figure for evaluation."""

    @property
    @abstractmethod
    def title(self) -> str:
        """The title (used in the fig.suptitle) of the figure."""
        pass

    @abstractmethod
    def get_figure(self, pred: np.ndarray, truth: np.ndarray, data_name: str, context: str) -> Figure:
        """
        Takes a set of predictions and truth values and some other data to generate a figure.

        Args:
            pred (np.ndarray): The set of predictions.
            truth (np.ndarray): The set of truth values.
            data_name (str): The data that the pred and truth represent.
            context (str): The context of the data (ex. "train", "test", etc)

        Returns:
            Figure: The generated figure.
        """
        pass


class MeanErrorHistogram(EvalFigure):
    @property
    def title(self) -> str:
        return "Mean Error Histogram"

    def get_figure(self, preds: np.ndarray, truths: np.ndarray, data_name: str, context: str) -> Figure:
        fig = plt.figure()

        mean_error = preds - truths

        std = float(np.std(mean_error))
        hist_range = (-std * 3, std * 3)
        plt.hist(mean_error, bins=100, density=True, range=hist_range)

        ax = plt.gca()

        ax.set_xlabel("PBLH (m)")
        ax.set_ylabel("Frequency")

        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

        value_label = units.get_formatted_units(DKey(data_name))

        if value_label is None:
            value_label = data_name

        ax = plt.gca()

        ax.set_xlabel(value_label)
        ax.set_ylabel("Frequency")

        plt.legend()

        fig.suptitle(f"{data_name} {self.title} ({context})")

        return fig


class EvalProfile(EvalFigure, ABC):
    @abstractmethod
    def get_profiles(self, pred: np.ndarray, truth: np.ndarray) -> Dict[str, np.ndarray]:
        pass

    def get_figure(self, pred: np.ndarray, truth: np.ndarray, data_name: str, context: str) -> Figure:
        fig = plt.figure()

        profiles = self.get_profiles(pred, truth)
        value_label = units.get_formatted_units(DKey(data_name))
        plot_profiles(profiles, value_axis=value_label)

        if len(profiles) > 1:
            plt.legend()

        fig.suptitle(f"{data_name} {self.title} ({context})")

        return fig


class MeanErrorProfile(EvalProfile):
    def __init__(self, absolute: bool, percentage: bool):
        self.absolute = absolute
        self.percentage = percentage

    @property
    def title(self) -> str:
        title = "Mean"

        if self.absolute:
            title += " Absolute"

        if self.percentage:
            title += " Percentage"

        return title + " Error"

    def get_profiles(self, pred: np.ndarray, truth: np.ndarray) -> Dict[str, np.ndarray]:
        profile = pred - truth

        if self.percentage:
            profile = (profile / truth) * 100

        if self.absolute:
            profile = np.abs(profile)

        profile = np.mean(profile, axis=0)
        return {self.title: profile}

    def get_figure(self, pred: np.ndarray, truth: np.ndarray, data_name: str, context: str) -> Figure:
        super_fig = super().get_figure(pred, truth, data_name, context)

        if self.percentage:
            super_fig.axes[0].set_xlabel("Error (%)")

        return super_fig


class VarCompProfile(EvalProfile):
    @property
    def title(self) -> str:
        return "Pred v. Truth Variance"

    def get_profiles(self, pred: np.ndarray, truth: np.ndarray) -> Dict[str, np.ndarray]:
        var_pred = np.var(pred, axis=0)
        var_truth = np.var(truth, axis=0)

        return {"Pred": var_pred, "Truth": var_truth}


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


def plot_eval_figures(
    preds: Dict[str, np.ndarray],
    truths: Dict[str, np.ndarray],
    eval_figures: List[EvalFigure],
    context: str,
    log: bool = False,
) -> List[Figure]:
    """
    Creates figures from EvalFigures with provided dictionaries of predictions and truths.

    Args:
        preds (Dict[str, np.ndarray]): A dictionary of string DKey to the array of predictions
        truths (Dict[str, np.ndarray]): A dictionary of string DKey to array of truths
        eval_figures (List[EvalFigure]): The list of MetricProfile to create profiles with
        context (str): The context of the preds and truths (ex. "train", "test", etc)
        log (bool, optional): Whether to log the figure to MLFlow. Defaults to False.

    Returns:
        List[Figure]: The list of generated figures.
    """
    figs = []

    for name in truths.keys():
        for eval_figure in eval_figures:
            if preds[name].ndim == 1 and isinstance(eval_figure, EvalProfile):
                continue

            if preds[name].ndim > 1 and not isinstance(eval_figure, EvalProfile):
                continue

            fig = eval_figure.get_figure(preds[name], truths[name], name, context)

            if log:
                mlflow_log.log_figure(fig, artifact_file=f"{context}_figures/{fig.get_suptitle()}.png")

            figs.append(fig)

    return figs


def get_eval_figs(
    model: Model,
    datasets: Dict[str, FullDaysDataset],
    eval_figures: List[EvalFigure],
    log: bool = False,
) -> List[Figure]:
    """
    Logs the useful evaluation profile plots for the train and test sets automatically.

    Currently plots the variance comparison and MAE per level for both the train and test datasets.

    Args:
        model (Model): A Keras model to predict from
        train (FullDaysDataset): The train dataset
        test (FullDaysDataset): The test dataset
    """
    figs = []

    for context, dataset in datasets.items():
        # Test profiles
        preds = dataset.predict(model, unscale=True)
        truths = dataset.get_targets(scaling=False)

        figs += plot_eval_figures(preds, truths, eval_figures, context, log=log)

    return figs


def log_metric_profile_figs(run_id: str, eval_figures: List[EvalFigure]) -> List[Figure]:
    """Takes an MLFlow run_id and a set of evaluation figures and automatically loads the model data
    and the datasets to create all of the requested figures and log them into MLFlow.

    Note that this assumes the MLFlow tracking_uri has been set same tracking directory as the run being used.

    Args:
        run_id (str): The id of the run.
        eval_figures (List[EvalFigure]): The list of evaluation figures to generate.

    Returns:
        List[Figure]: The list of generated Figures (note that they are automatically logged to MLFlow!)
    """
    datasets = get_datasets_from_run(run_id)
    model = mlflow_log.get_autolog_model(run_id)

    return get_eval_figs(model, datasets, eval_figures, log=True)
