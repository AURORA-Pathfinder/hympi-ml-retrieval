"""A useful set of utility functions that wrap around common MLFlow functionality."""

from typing import Dict, Optional

import mlflow
import mlflow.entities
from mlflow.tracking._tracking_service.client import TrackingServiceClient

import keras.models
from matplotlib.figure import Figure


def start_run(tracking_uri: str, experiment_name: str, **autolog_args) -> mlflow.ActiveRun:
    """
    Starts a new mlflow run with a given experiment name and returns the ActiveRun.

    Args:
        tracking_uri (str): The tracking uri that points to the mlruns directory containing runs.
            Could be an address where the mlruns are hosted or a direct file path.
        experiment_name (str): The name of the experiment to start the run in.
        **autolog_args (kwargs): A set of keyword arguments applied to the mlflow.autolog function.

    Returns:
        mlflow.ActiveRun: _description_
    """
    mlflow.set_tracking_uri(f"file://{tracking_uri}")

    mlflow.set_experiment(experiment_name)

    mlflow.autolog(**autolog_args)
    return mlflow.start_run()


def get_artifacts_uri(run_id: str) -> str:
    """
    Returns the absolute path of the artifacts directory given a run_id.

    Args:
        run_id (str): The id of the mlflow run

    Returns:
        str: The absolute path of the artifacts directory for the mlflow run
    """
    client = TrackingServiceClient(mlflow.get_tracking_uri())
    return client._get_artifact_repo(run_id).artifact_uri


def get_autolog_model(run_id: str) -> keras.models.Model:
    """Returns the Keras model that is logged by MLFlow when autologging models is enabled

    Args:
        run_id (str): The id of the mlflow run.

    Returns:
        keras.models.Model: The loaded Keras model.
    """
    return get_model(run_id, "model/data/model")


def get_model(run_id: str, local_artifact_path: str) -> keras.models.Model:
    """Returns a keras model located in the artifacts of the run at the local artifact path.

    Args:
        run_id (str): The id of the mlflow run.
        local_artifact_path (str): The local path of the model in the artifacts directory of the mlflow run.

    Returns:
        keras.models.Model: The loaded Keras model.
    """
    artifacts_uri = get_artifacts_uri(run_id)
    model_path = f"{artifacts_uri}/{local_artifact_path}"
    return keras.models.load_model(model_path)


def get_datasets_by_context(run_id: str) -> Dict[str, mlflow.data.Dataset]:
    """
    Given a run_id, this function gathers the datasets in an mlflow run and
    creates a dictionary with dataset's context tag as the key and the generic mflow dataset
    itself as the value.

    Args:
        run_id (str): The run id of the mlflow run

    Returns:
        Dict[str, mlflow.data.Dataset]: The dictionary whos keys are the context tag of the dataset value
    """
    run = mlflow.get_run(run_id)

    dataset_dict = {}

    for dataset_input in run.inputs.dataset_inputs:
        context = "unknown"

        for tag in dataset_input.tags:
            if tag.key == "mlflow.data.context":
                context = tag.value
        dataset_dict.update({context: dataset_input.dataset})

    return dataset_dict


def log_figure(fig: Figure, artifact_file: Optional[str] = None):
    """
    Logs a figure to mlflow. This function wraps the standanrd log_figure function in mlflow but
    makes the file path optional as it is automatically calculates based on the title of the figure.

    Args:
        fig (Figure): The figure to log.
        artifact_file (Optional[str], optional): The relative artifact file path where the figure is saved.
            If None, will be placed in the main artifact directory as a png with the title of the figure.
            Defaults to None.
    """
    if artifact_file is None:
        artifact_file = f"{fig.get_suptitle()}.png"

    mlflow.log_figure(fig, artifact_file)
