from typing import Literal, Mapping

import torch
from torch import nn
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from torchmetrics import Metric, MetricCollection
import numpy as np
import mlflow

from hympi_ml.data import DataSpec


class SpecModel(L.LightningModule):
    def __init__(
        self,
        features: Mapping[str, DataSpec],
        targets: Mapping[str, DataSpec],
        train_metrics: Mapping[str, MetricCollection],
        val_metrics: Mapping[str, MetricCollection],
        test_metrics: Mapping[str, MetricCollection],
        learning_rate: float = 0.001,
        weight_decay: float = 0,
    ):
        super().__init__()

        # important! Used to ensure all above init params are saved!
        self.save_hyperparameters()

        self.features = features
        self.targets = targets
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def transform_batch(self, raw_batch):
        """
        Takes in a raw batch as a tuple of (features, targets) and, based on
        this model's DataSpec's, will apply the correct transformations and
        filtering as needed.

        NOTE: The the filtering is applied *before* scaling!
        """
        features_batch, targets_batch = raw_batch

        # calculate filter mask
        final_mask = None

        for key, feature in features_batch.items():
            mask = self.features[key].get_filter_mask(feature)
            final_mask = (final_mask & mask) if (final_mask is not None) else mask

        for key, target in targets_batch.items():
            mask = self.targets[key].get_filter_mask(target)
            final_mask = (final_mask & mask) if (final_mask is not None) else mask

        # apply batch transformations based on scale_ranges
        transformed_features = {
            k: self.features[k].apply_batch(features_batch[k][final_mask])
            for k in self.features.keys()
        }

        transformed_targets = {
            k: self.targets[k].apply_batch(targets_batch[k][final_mask])
            for k in self.targets.keys()
        }

        # return the transformed features
        return transformed_features, transformed_targets

    def calculate_metrics(
        self, raw_batch: torch.Tensor, metrics: Mapping[str, MetricCollection]
    ):
        inputs, targets = self.transform_batch(raw_batch)
        outputs = self(inputs)

        computed_metrics = {}

        for k in self.targets.keys():
            target_metrics = metrics[k]
            target_metrics.prefix = f"{target_metrics.prefix}{k}_"
            computed_metrics.update(target_metrics(outputs[k], targets[k]))

        return computed_metrics

    def training_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, self.train_metrics)
        metrics["loss"] = torch.vstack(list(metrics.values())).sum()

        self.log_dict(metrics, prog_bar=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, self.val_metrics)
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch, self.test_metrics)
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


class MLPModel(SpecModel):
    def __init__(
        self,
        features: Mapping[str, DataSpec],
        targets: Mapping[str, DataSpec],
        train_metrics: Mapping[str, MetricCollection],
        val_metrics: Mapping[str, MetricCollection],
        test_metrics: Mapping[str, MetricCollection],
        feature_paths: dict[str, nn.Module],
        output_path: nn.Module,
    ):
        super().__init__(features, targets, train_metrics, val_metrics, test_metrics)

        # important! Used to ensure all above init params are saved!
        self.save_hyperparameters()

        self.feature_paths = feature_paths
        self.output_path = output_path

    def forward(self, inputs):
        # go through all feature paths for each feature and combine them into one tensor
        features_forward = [
            path.cuda()(inputs[k].cuda()) for k, path in self.feature_paths.items()
        ]
        stacked_features = torch.hstack(features_forward)

        # run stacked features tensor into the output path for each target
        return {k: self.output_path(stacked_features) for k in self.targets.keys()}
