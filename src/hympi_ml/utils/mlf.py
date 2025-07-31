"""A useful set of utility functions that wrap around common MLFlow functionality."""

import os
import glob

import mlflow
from mlflow.tracking._tracking_service.client import TrackingServiceClient

from matplotlib.figure import Figure


def start_run(
    tracking_uri: str, experiment_name: str, **autolog_args
) -> mlflow.ActiveRun:
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


def get_checkpoint_path(run_id: str, step: int | None = None) -> str:
    """Returns the path of the PyTorch LightningModule checkpoint at the provided step.
    If no step is provided, then it will find the best or latest checkpoint saved in this run.

    Args:
        run_id (str): The id of the mlflow run.
        step (int): The step the checkpoint was saved at (often in the title of the checkpoint file itself).

    Returns:
        str: The absolute file path to the checkpoint file.
    """
    artifacts_uri = get_artifacts_uri(run_id)[7:]

    best_path = f"{artifacts_uri}/checkpoints/latest_checkpoint.pth"
    if os.path.exists(best_path):
        return best_path

    step_str = step or "*"
    files = glob.glob(f"{artifacts_uri}/*/epoch=*-step={step_str}*.ckpt")
    return sorted(files)[0]


def log_figure(
    fig: Figure,
    artifact_path: str,
    run_id: str | None = None,
):
    """
    Logs a figure to mlflow. This function wraps the standanrd log_figure function in mlflow but
    makes the file path optional as it is automatically calculates based on the title of the figure.

    Args:
        fig (Figure): The figure to log.
        artifact_path (str): The path in the artifacts file where the file is saved.
        run_id (str | None): The run_id of the run to log the figure to. If None, will log to the current run.
    """
    local_path = f"/tmp/{os.path.basename(artifact_path)}"
    fig.savefig(local_path)
    mlflow.log_artifact(local_path, f"{os.path.dirname(artifact_path)}", run_id=run_id)
