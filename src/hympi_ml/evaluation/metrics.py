import torch
from torchmetrics import Metric
import matplotlib.pyplot as plt


class ErrorHistogram(Metric):
    def __init__(
        self, num_bins: int = 500, min: int = -2000, max: int = 2000, **kwargs
    ):
        super().__init__(**kwargs)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_bins = num_bins
        self.min = min
        self.max = max
        self.bin_edges = torch.linspace(self.min, self.max, num_bins, device=device)

        bin_counts = torch.zeros(self.num_bins, dtype=torch.long, device=device)
        self.add_state("bin_counts", default=bin_counts, dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        errors = torch.subtract(preds, target).flatten()

        clamped_errors = torch.clamp(errors, min=self.min, max=self.max - 1e-6)

        bin_indices = torch.bucketize(clamped_errors, self.bin_edges)
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        current_batch_counts = torch.bincount(bin_indices, minlength=self.num_bins)

        self.bin_counts += current_batch_counts

    def plot(self):
        bar_width = (self.max - self.min) / self.num_bins
        plt.bar(
            self.bin_edges.cpu().numpy(),
            height=self.compute().cpu().numpy(),
            width=bar_width,
        )

    def compute(self):
        return self.bin_counts
