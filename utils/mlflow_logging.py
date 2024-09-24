from typing import Dict, Any

import mlflow

mlruns_base = "/data/nature_run/hympi-ml-retrieval/mlruns"
mlflow.set_tracking_uri(f"file://{mlruns_base}")

# starts a new mlflow run with a given experiment name and returns the run_id
def start_run(experiment_name: str, **autolog_args: Dict[str, Any]) -> mlflow.ActiveRun:
    '''
    Starts a new mlflow run with a given experiment name and returns the ActiveRun.
    '''
    mlflow.set_experiment(experiment_name)

    mlflow.autolog(**autolog_args)
    return mlflow.start_run()

def end_run():
    mlflow.end_run()
