from typing import Any, Dict

import mlflow
from keras.models import Model

from hympi_ml.data.memmap import MemmapBatches


def log_eval_metrics(model: Model, batches: MemmapBatches, context: str) -> Dict[str, Any]:
    """
    Runs `keras.model.evaluate` on the given model with the given MemmapBatches. Logs the output set of metrics
    on MLFlow prefixed with the given context string.

    Returns the output dictionary of evaluation metrics.
    """
    eval_metrics: Dict[str, Any] = model.evaluate(batches, verbose=1, return_dict=True)

    for key in eval_metrics.keys():
        mlflow.log_metric(f"{context}_{key}", eval_metrics[key])

    return eval_metrics
