from typing import List, Tuple, Dict, Any
import mlflow
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


def log_dataset(feature_shapes: List[Tuple], target_shape: Tuple, dataset_name: str, context: str):
    features_dict: Dict[str, np.ndarray] = {}
    for i in range(len(feature_shapes)):
        features_dict[str(i)] = np.empty(feature_shapes[i])
    
    targets = np.empty(target_shape)

    dataset = mlflow.data.from_numpy(features=features_dict, targets=targets, name=dataset_name)    
    mlflow.log_input(dataset, context=context)


def end_run():
    mlflow.end_run()
