from typing import Any, Dict

import mlflow
from keras.models import Model
import numpy as np

from data.memmap import MemmapBatches


def log_eval_metrics(
    model: Model, batches: MemmapBatches, context: str
) -> Dict[str, Any]:
    """
    Runs `keras.model.evaluate` on the given model with the given MemmapBatches. Logs the output set of metrics
    on MLFlow prefixed with the given context string.

    Returns the output dictionary of evaluation metrics.
    """
    eval_metrics: Dict[str, Any] = model.evaluate(batches, verbose=1, return_dict=True)

    for key in eval_metrics.keys():
        mlflow.log_metric(f"{context}_{key}", eval_metrics[key])

    return eval_metrics


def var_per_level(vals: np.array):
    levels = vals.shape[1]
    return [np.var(vals[:, i]) for i in range(levels)]


def mae_per_level(pred: np.ndarray, truth: np.ndarray):
    levels = pred.shape[1]
    return [np.average(np.abs(pred[:, i] - truth[:, i])) for i in range(levels)]
