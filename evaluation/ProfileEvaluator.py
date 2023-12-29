import os

from numpy import ndarray
from evaluation.Evaluator import Evaluator
import matplotlib.pyplot as plt
import mlflow

import numpy as np

import netCDF4

# General evaluator for any kind of profile
class ProfileEvaluator(Evaluator):
    def __init__(self, show_plot: bool, net_cdf: str):
        self.show_plot = show_plot

        if net_cdf:
            self.nc_gen = True
            self.nc_path = net_cdf

    def evaluate(self, pred: ndarray, truth: ndarray, latlon: ndarray,
                 name_prefix: str, show_plot: bool):
        super().evaluate(pred, truth, name_prefix)

        self.log_mae_per_level(pred, truth, name_prefix)
        self.log_pred_vs_true(pred, truth, index=0, name_prefix=name_prefix)

        if self.nc_gen:
            self.generate_netcdf(pred, truth, latlon, name_prefix)

    def log_mae_per_level(self, pred: ndarray, truth: ndarray, name_prefix: str):
        error = pred - truth
        error_fig = plt.figure()
        plt.plot([np.average(np.abs(error[:,i])) for i in range(72)])
        if self.show_plot:
            plt.show()

        mlflow.log_figure(error_fig, name_prefix + "_mae_per_level.png")

    def log_pred_vs_true(self, pred: ndarray, truth: ndarray, index: int, name_prefix: str):
        base_dir = "temp/"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        npy_file_path = base_dir + name_prefix + "_pred_(index " + str(index) + ").npy"
        np.save(npy_file_path, pred[index])
        mlflow.log_artifact(npy_file_path)

        pred_vs_true_fig = plt.figure()
        plt.plot(truth[index], label="true")
        plt.plot(pred[index], label="pred")
        plt.legend(loc="lower right")
        if self.show_plot:
            plt.show()

        mlflow.log_figure(pred_vs_true_fig, name_prefix + "_pred_vs_true_(index " + str(index) + ").png")

    def generate_netcdf(self, pred: ndarray, truth: ndarray, latlon: ndarray, name_prefix: str):
        # TODO: Implement this!
        # TODO: Surface pressure
        fname = f"{name_prefix}_{self.config_name}_{self.current_run}.nc"
        fn = f"{self.nc_path}/{fname}"

        with netCDF4.Dataset(fn, 'w', format='NETCDF4') as nc:

            num_rows = latlon.shape[0]

            nc.title = "Truth vs Predicted Profiles"
            nc.run_config = self.config_name
            nc.surface_pressure = 1000.0

            rows_dim = nc.createDimension('row', num_rows)
            # rows_dim.units = "Index"
            # rows_dim.long_name = "Sample Index in the data set"

            z_dim = nc.createDimension('z', 72)
            # z_dim.units = "Sigma"
            # z_dim.long_name = "Pressure level in Sigma"

            z_true = nc.createVariable('profile_true', np.float64, ('row', 'z'))
            z_pred = nc.createVariable('profile_pred', np.float64, ('row', 'z'))
            lat = nc.createVariable('lat', np.float64, ('row'))
            lon = nc.createVariable('lon', np.float64, ('row'))

            lat[:] = latlon[:, 0]
            lon[:] = latlon[:, 1]
            z_true[:, :] = truth
            z_pred[:, :] = pred
