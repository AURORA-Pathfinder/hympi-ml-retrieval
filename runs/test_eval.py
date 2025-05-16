import os

os.environ["KERAS_BACKEND"] = "torch"


import mlflow
from hympi_ml.utils import mlflow_log

import hympi_ml.data.model_dataset as md

from hympi_ml.data.ch06 import Ch06Source

mlflow.set_tracking_uri("/home/dgershm1/mlruns")

run_id = "2806d0984b9044e487f5614185fb16a6"
datasets = md.get_datasets_from_run(run_id)
model = mlflow_log.get_autolog_model(run_id)

datasets['test'].evaluate(model, metrics=["mae", "mse"], unscale=True, context="test")