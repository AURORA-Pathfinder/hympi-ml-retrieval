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
        features_forward = [
            path.cuda()(self.features[k].apply_batch(inputs[k]))
            for k, path in self.feature_paths.items()
        ]
        return self.output_path(torch.concat(features_forward))

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        output = self(inputs)
        return nn.functional.mse_loss(output, targets["TEMPERATURE"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


features = {"AMPR": AMPRSpec()}
targets = {"TEMPERATURE": NRSpec(dataset="TEMPERATURE")}

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
    # "AMPR": AMPRSpec(),
}

model = SpecModel(
    features=features,
    targets=targets,
    feature_paths={
        "CH_16": nn.Sequential(
            nn.LazyLinear(96),
            nn.GELU(),
        )
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

dataset = ModelDataset(
    Ch06Source(
        days=[
            "20060115",
            # "20060215",
            # "20060315",
            # "20060415",
            # "20060515",
            # "20060615",
            # "20060715",
        ]
    ),
    features=features,
    targets=targets,
    batch_size=8192,
)

loader = DataLoader(
    dataset,
    batch_size=None,
    shuffle=True,
    num_workers=40,
    pin_memory=True,
    prefetch_factor=1,
)

trainer = L.Trainer(enable_progress_bar=True, max_epochs=2, enable_model_summary=True)
trainer.fit(model=model, train_dataloaders=loader)
