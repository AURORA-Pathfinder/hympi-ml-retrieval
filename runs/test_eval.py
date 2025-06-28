import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
from torch import Tensor

import keras
from keras import ops

import mlflow
from hympi_ml.utils import mlflow_log

import hympi_ml.data.model_dataset as md

# from hympi_ml.data import RFBand
# import hympi_ml.data.cosmirh as c
from hympi_ml.data.ch06 import Ch06Source
from hympi_ml.evaluation.metrics import (
    MeanErrorProfile,
    VarianceErrorProfile,
    # MeanErrorHistogram,
)

mlflow.set_tracking_uri("/home/dgershm1/mlruns")

run_id = "703619f453124fdeaf2015954984c153"
datasets = md.get_datasets_from_run(run_id)
model = mlflow_log.get_autolog_model(run_id)

test = datasets["test"]

test.batch_size = 4096 * 2

print(test[0])


class ScalarErrorSet(keras.Metric):
    """
    Creates an array containing every single error.
    """

    def __init__(self, **kwargs):
        super().__init__(name="scalar_error_set", **kwargs)
        self.range = None
        self.all_values = self.add_variable(
            name="all_values",
            initializer=keras.initializers.Zeros(),
            shape=(0,),  # Start with an empty shape
        )
        self._current_values_list = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = ops.subtract(y_true, ops.reshape(y_pred, newshape=y_true.shape))

        error = ops.reshape(error, (-1,))

        self._current_values_list

    def reset_state(self):
        self.set = None

    def result(self):
        return self.set


class ErrorHistogram(keras.Metric):
    def __init__(
        self, num_bins: int = 500, min: int = -2000, max: int = 2000, **kwargs
    ):
        super().__init__(name="error_histogram", **kwargs)
        self.num_bins = num_bins
        self.min = min
        self.max = max
        self.bin_edges = torch.linspace(self.min, self.max, num_bins + 1).cuda()
        self.reset_state()

    def reset_state(self):
        self.bin_counts = torch.zeros(self.num_bins, dtype=torch.long).cuda()
        self.total_samples = 0

    def update_state(self, y_true, y_pred, sample_weight):
        y_pred = y_pred.view(-1).cuda()
        y_true = y_true.view(-1).cuda()

        errors = torch.subtract(y_pred, y_true)

        clamped_errors = torch.clamp(errors, min=self.min, max=self.max - 1e-6)

        bin_indices = torch.bucketize(clamped_errors, self.bin_edges)
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        current_batch_counts = torch.bincount(bin_indices, minlength=self.num_bins)

        self.bin_counts += current_batch_counts

    def result(self):
        return self.bin_counts


with mlflow.start_run(run_id=run_id):
    evals = test.evaluate(
        model,
        metrics={
            "PBLH": [ErrorHistogram()],
        },
        context="test",
        unscale=True,
        log=True,
    )

print(evals)
