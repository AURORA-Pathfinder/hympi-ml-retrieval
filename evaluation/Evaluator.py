from mlflow import ActiveRun, log_metric
import numpy as np

# the base evaluate class, computes some basic evaluation metrics
class Evaluator():
    def set_current_run(self, current_run: ActiveRun = None):
        self.current_run = current_run

    # performs any kind of evaluation based on predicted values
    # By default, this performs a simple MAE and MSE metric calculation and logs it
    # optional "name_prefix" variable for logging purposes (adds a prefix to the logged evaluation metrics to separate it from other evaluate runs)
    def evaluate(self, pred: np.ndarray, truth: np.ndarray, name_prefix: str = ""):

        error = pred - truth

        mae = np.average(np.abs(error))
        log_metric(name_prefix + "_mae", mae)

        mse = np.average(np.square(error))
        log_metric(name_prefix + "_mse", mse)