import os

os.environ["KERAS_BACKEND"] = "torch"

import keras

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader

from hympi_ml.data import (
    DataSpec,
    DataSource,
    CosmirhSpec,
    AMPRSpec,
    NRSpec,
    ch06,
    RFBand,
)

from hympi_ml.data.cosmirh import C50_BAND, C183_BAND
from hympi_ml.data.ch06 import Ch06Source
from hympi_ml.data.model_dataset import ModelDataset


class SpecModel(L.LightningModule):
    def __init__(
        self,
        features: dict[str, DataSpec],
        targets: dict[str, DataSpec],
        feature_paths: dict[str, nn.Module],
        output_path: nn.Module,
        loss,
    ):
        super().__init__()
        self.features = features
        self.targets = targets
        self.feature_paths = feature_paths
        self.output_path = output_path
        self.loss = loss

    def forward(self, inputs):
        # go through all feature paths for each feature and combine them into one tensor
        features_forward = [
            path.cuda()(self.features[k].apply_batch(inputs[k]))
            for k, path in self.feature_paths.items()
        ]
        stacked_features = torch.hstack(features_forward)

        # run stacked features tensor into the output path for each target
        return {
            k: nn.LazyLinear(target.shape[0]).cuda()(self.output_path(stacked_features))
            for k, target in targets.items()
        }

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        output = self(inputs)

        losses = {
            k: nn.functional.l1_loss(output[k], self.targets[k].apply_batch(targets[k]))
            for k in self.targets.keys()
        }

        losses["loss"] = torch.vstack(list(losses.values())).sum()

        self.log_dict(losses, prog_bar=True)

        return losses

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


features = {
    "CH_16": CosmirhSpec(
        frequencies=[
            C50_BAND.scale(16),
            C183_BAND.scale(16),
        ],
        ignore_frequencies=[  # problematic CRTM frequencies
            56.96676875,
            57.60739375,
            57.6113,
            57.61520625,
        ],
        scale_range=(200, 300),
    ),
    "AMPR": AMPRSpec(),
}

targets = {
    "TEMPERATURE": NRSpec(
        dataset="TEMPERATURE",
        scale_range=(175.0, 325.0),
    ),
    "WATER_VAPOR": NRSpec(
        dataset="WATER_VAPOR",
        scale_range=(1.11e-7, 3.08e-2),
    ),
}

model = SpecModel(
    features=features,
    targets=targets,
    feature_paths={
        "CH_16": nn.Sequential(
            nn.LazyLinear(96),
            nn.GELU(),
        ),
        "AMPR": nn.Sequential(
            nn.LazyLinear(8),
            nn.GELU(),
        ),
    },
    output_path=nn.Sequential(
        nn.LazyLinear(128),
        nn.GELU(),
        nn.LazyLinear(128),
        nn.GELU(),
        nn.LazyLinear(72),
    ),
    loss=nn.MSELoss(),
)

train_dataset = ModelDataset(
    Ch06Source(
        days=[
            "20060115",
            "20060215",
            # "20060315", # removed due to space constraints
            "20060415",
            "20060515",
            "20060615",
            "20060715",
            "20061015",
            "20061115",
        ]
    ),
    features=features,
    targets=targets,
    batch_size=8192,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=None,
    shuffle=True,
    num_workers=40,
    pin_memory=True,
    prefetch_factor=1,
)

trainer = L.Trainer(enable_progress_bar=True, max_epochs=2, enable_model_summary=True)
trainer.fit(model=model, train_dataloaders=train_loader)


test_dataset = ModelDataset(
    Ch06Source(days=["20060815"]),
    features=features,
    targets=targets,
    batch_size=8192,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=None,
    shuffle=True,
    num_workers=40,
    pin_memory=True,
    prefetch_factor=1,
)

# trainer.test(model=model, dataloaders=test_loader)
