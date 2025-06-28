from typing import Literal

from keras import Metric
from keras import ops

import torch


class TensorMetric(Metric):
    """
    A wrapper class to be considered a base for any metric that returns a tensor as its result.
    """

    pass


class MeanErrorProfile(TensorMetric):
    """
    Creates a profile tensor containing the mean error (among other operations as needed) for each sigma level.
    """

    def __init__(
        self,
        levels: int = 72,
        other_ops: Literal["absolute", "squared", "root_squared"] | None = None,
        percentage: bool = False,
        **kwargs,
    ):
        self.other_ops = other_ops
        self.percentage = percentage

        name = "mean"

        match other_ops:
            case "absolute" | "squared":
                name += "_" + other_ops
            case "root_squared":
                name = "root_mean_squared"

        if percentage:
            name += "_percentage"

        name += "_error"

        super().__init__(name=name, **kwargs)
        self.total = self.add_variable(
            shape=(levels,), initializer="zeros", dtype=self.dtype, name="total"
        )
        self.count = self.add_variable(
            shape=(), initializer="zeros", dtype=self.dtype, name="count"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = ops.subtract(y_true, y_pred)

        if self.percentage:
            error = ops.divide(error, y_true)

        match self.other_ops:
            case "absolute":
                error = ops.absolute(error)
            case "squared" | "root-squared":
                error = ops.square(error)

        if len(error.shape) >= 1:
            num_samples = ops.shape(error)[0]
            self.total.assign_add(ops.sum(error, axis=0))
        else:
            num_samples = 1
            self.total.assign_add(error)

        self.count.assign_add(ops.cast(num_samples, dtype=self.dtype))

    def reset_state(self):
        self.total.assign(ops.zeros(shape=self.total.shape))
        self.count.assign(0)

    def result(self):
        div = ops.divide_no_nan(self.total, ops.cast(self.count, dtype=self.dtype))

        if self.other_ops == "root_squared":
            return ops.sqrt(div)

        return div


class VarianceErrorProfile(TensorMetric):
    """
    Creates a profile tensor containing the variance error for each sigma level.
    """

    def __init__(self, levels: int = 72, **kwargs):
        super().__init__(name="variance_error", **kwargs)
        self.total = self.add_variable(
            shape=(levels,), initializer="zeros", dtype=self.dtype, name="total"
        )
        self.count = self.add_variable(
            shape=(), initializer="zeros", dtype=self.dtype, name="count"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_var = ops.var(y_true, axis=0)
        pred_var = ops.var(y_pred, axis=0)
        var_error = ops.subtract(pred_var, true_var)
        var_error = ops.divide(var_error, true_var)

        if len(y_true.shape) >= 1:
            num_samples = ops.shape(y_true)[0]
            self.total.assign_add(ops.sum(var_error, axis=0))
        else:
            num_samples = 1
            self.total.assign_add(var_error)

        self.count.assign_add(ops.cast(num_samples, dtype=self.dtype))

    def reset_state(self):
        self.total.assign(ops.zeros(shape=self.total.shape))
        self.count.assign(0)

    def result(self):
        return ops.divide_no_nan(self.total, ops.cast(self.count, dtype=self.dtype))


class ErrorHistogram(TensorMetric):
    def __init__(
        self, num_bins: int = 500, min: int = -2000, max: int = 2000, **kwargs
    ):
        super().__init__(
            name=f"error_histogram_{num_bins}-bins_({min},{max})", **kwargs
        )
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
