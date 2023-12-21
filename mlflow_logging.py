from typing import Dict, Any
import mlflow
from preprocessing.ModelIO import ModelIO
import numpy as np

# TODO set this with config var
mlruns_base = "/data/nature_run/hympi-ml-retrieval/mlruns"
mlflow.set_tracking_uri(f"file://{mlruns_base}")


# starts a new mlflow run with a given experiment name and returns the run_id
def start_run(experiment_name: str, **autolog_args: Dict[str, Any]) -> mlflow.ActiveRun:
    mlflow.set_experiment(experiment_name)

    mlflow.autolog(**autolog_args)
    return mlflow.start_run().info.run_id


def log_config(config_path: str, config_name: str):
    mlflow.log_artifact(config_path + "/" + config_name + ".yaml")


# replaces the name of the first input dataset to the given name in a given active run
def log_dataset(dataset_name: str, context: str, data: ModelIO):
    features_data = data.get_transformed_features()

    if (len(data.features) > 1):
        features_dict: Dict[str, np.ndarray] = {}
        for i in range(len(features_data)):
            features_dict[str(i)] = features_data[i]
        features_data = features_dict

    dataset: mlflow.data.Dataset = mlflow.data.from_numpy(features=features_data,
                                                          targets=data.get_transformed_target(),
                                                          name=dataset_name)
    mlflow.log_input(dataset, context=context)


def end_run():
    mlflow.end_run()
