import torch
from torchmetrics import Metric


class ErrorHistogram(Metric):
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
        self.bin_counts = torch.zeros(self.num_bins, dtype=torch.long).cuda()
        self.total_samples = 0

    def update(self, preds, target):
        preds = preds.view(-1).cuda()
        target = target.view(-1).cuda()

        errors = torch.subtract(preds, target)

        clamped_errors = torch.clamp(errors, min=self.min, max=self.max - 1e-6)

        bin_indices = torch.bucketize(clamped_errors, self.bin_edges)
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        current_batch_counts = torch.bincount(bin_indices, minlength=self.num_bins)

        self.bin_counts += current_batch_counts

    def compute(self):
        return self.bin_counts
