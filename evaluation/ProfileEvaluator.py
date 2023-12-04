from numpy import ndarray
from evaluation.Evaluator import Evaluator
import matplotlib.pyplot as plt
import mlflow

import numpy as np

# General evaluator for any kind of profile
class ProfileEvaluator(Evaluator):
    def evaluate(self, pred: ndarray, truth: ndarray, name_prefix: str):
        super().evaluate(pred, truth, name_prefix)

        self.log_mae_per_level(pred, truth, name_prefix)
        self.log_pred_vs_true(pred, truth, index=0, name_prefix=name_prefix)

    def log_mae_per_level(self, pred: ndarray, truth: ndarray, name_prefix: str):
        error = pred - truth
        error_fig = plt.figure()
        plt.plot([np.average(np.abs(error[:,i])) for i in range(72)])
        plt.show()

        mlflow.log_figure(error_fig, name_prefix + "_mae_per_level.png")

    def log_pred_vs_true(self, pred: ndarray, truth: ndarray, index: int, name_prefix: str):

        npy_file_path = "temp/" + name_prefix + "_pred_(index " + str(index) + ").npy"
        np.save(npy_file_path, pred[index])
        mlflow.log_artifact(npy_file_path)

        pred_vs_true_fig = plt.figure()
        plt.plot(truth[index], label="true")
        plt.plot(pred[index], label="pred")
        plt.legend(loc="lower right")
        plt.show()

        mlflow.log_figure(pred_vs_true_fig, name_prefix + "_pred_vs_true_(index " + str(index) + ").png")

    
    def generate_netcdf(self):
        # TODO: Implement this!
        pass