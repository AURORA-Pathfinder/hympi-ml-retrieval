import mlflow
import numpy as np

def log_mae(error: np.ndarray, name: str):
    mae = np.average(np.abs(error))
    mlflow.log_metric(f"{name}_mae", mae)

def log_mse(error: np.ndarray, name: str):
    mse = np.average(np.square(error))
    mlflow.log_metric(f"{name}_mse", mse)