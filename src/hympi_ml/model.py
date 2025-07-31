from typing import Literal, Mapping, Any

import torch
from torch import nn
import lightning as L
from torchmetrics import MetricCollection

from hympi_ml.data import ModelDataSpec


class SpecModel(L.LightningModule):
    def __init__(
        self,
        spec: ModelDataSpec,
        train_metrics: Mapping[str, MetricCollection],
        val_metrics: Mapping[str, MetricCollection],
        test_metrics: Mapping[str, MetricCollection],
    ):
        super().__init__()

        # important! Used to ensure all above init params are saved!
        self.save_hyperparameters()

        self.spec = spec
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

        self.log_metrics = True
        self._unscale_metrics = False
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._optimizer = None

    @property
    def unscale_metrics(self):
        return self._unscale_metrics

    @unscale_metrics.setter
    def unscale_metrics(self, value: bool):
        """
        If true, unscales the target values of a batch when calculating metrics.
        This is useful for test analysis and is not reccommended for training use.
        """
        self._unscale_metrics = value

    def calculate_metrics(
        self,
        raw_batch: torch.Tensor,
        metrics: Mapping[str, MetricCollection],
        context: Literal["train", "val", "test"],
    ):
        inputs, targets, _ = self.spec.transform_batch(raw_batch)
        outputs = self(inputs)

        if self._unscale_metrics:
            targets = self.spec.unscale_targets(targets)
            outputs = self.spec.unscale_targets(outputs)

        computed_metrics = {}

        for k in self.spec.targets.keys():
            target_metrics = metrics[k].to(device=self._device)
            target_metrics.prefix = f"{context}_{k}_"
            computed_metrics.update(target_metrics(outputs[k], targets[k]))

        return computed_metrics

    def training_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, self.train_metrics, "train")

        loss = torch.vstack(list(metrics.values())).sum()
        self.log("loss", loss)

        if self.log_metrics:
            self.log_dict(metrics, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, self.val_metrics, "val")

        val_loss = torch.vstack(list(metrics.values())).sum()
        self.log("val_loss", val_loss)

        if self.log_metrics:
            self.log_dict(metrics, prog_bar=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, self.test_metrics, "test")

        if self.log_metrics:
            self.log_dict(metrics, prog_bar=True)

    def set_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        | dict[str, Any]
        | None = None,
    ):
        if lr_scheduler is None:
            self._optimizer = optimizer
        else:
            self._optimizer = {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            }

    def configure_optimizers(self):
        if self._optimizer is None:
            return torch.optim.Adam(self.parameters())

        return self._optimizer


class MLPModel(SpecModel):
    def __init__(
        self,
        spec: ModelDataSpec,
        train_metrics: Mapping[str, MetricCollection],
        val_metrics: Mapping[str, MetricCollection],
        test_metrics: Mapping[str, MetricCollection],
        feature_paths: nn.ModuleDict,
        output_path: nn.Module,
    ):
        super().__init__(spec, train_metrics, val_metrics, test_metrics)

        # important! Used to ensure all above init params are saved!
        self.save_hyperparameters()

        self.feature_paths = feature_paths
        self.output_paths = nn.ModuleDict(
            {
                k: nn.Sequential(
                    output_path,
                    nn.LazyLinear(self.spec.targets[k].shape[0]),
                )
                for k in self.spec.targets.keys()
            }
        )

    def forward(self, inputs):
        # go through all feature paths for each feature and combine them into one tensor
        features_forward = [
            path.to(device=self._device)(inputs[k])
            for k, path in self.feature_paths.items()
        ]
        stacked_features = torch.hstack(features_forward)

        # run stacked features tensor into the output path for each target
        return {
            k: self.output_paths[k](stacked_features) for k in self.spec.targets.keys()
        }
