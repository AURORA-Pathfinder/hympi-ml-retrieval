import lightning as L
import lightning.pytorch.callbacks as callbacks
from lightning.pytorch.loggers import MLFlowLogger
import torch
from torch import nn
from torchmetrics import MetricCollection
import torchmetrics.regression as re

from hympi_ml.data import (
    ModelDataSpec,
    RawDataModule,
    cosmirh,
    CosmirhSpec,
    AMPRSpec,
    NRSpec,
)

from hympi_ml.data.scale import MinMaxScaler

from hympi_ml.data.ch06 import Ch06Source
from hympi_ml.model import MLPModel
from hympi_ml.utils import mlf

# data specification
spec = ModelDataSpec(
    features={
        # "CH": CosmirhSpec(
        #     frequencies=[
        #         cosmirh.C50_BAND,
        #         cosmirh.C183_BAND,
        #     ],
        #     ignore_frequencies=[  # problematic CRTM frequencies
        #         56.9667968,
        #         57.6074218,
        #         57.611328,
        #         57.6152343,
        #     ],
        # ),
        "AMPR": AMPRSpec(),
    },
    targets={
        "TEMPERATURE": NRSpec(
            dataset="TEMPERATURE",
            scaler=MinMaxScaler(minimum=175.0, maximum=345.0),
        ),
    },
)

# define data module (train, val, test)
datamodule = RawDataModule(
    spec=spec,
    train_source=Ch06Source(
        days=[
            "20060115",
            # "20060215",
            # "20060415",
            # "20060515",
            # "20060615",
            # "20060715",
            # "20060815",
            # "20061015",
        ]
    ),
    val_source=Ch06Source(days=["20061115"]),
    test_source=Ch06Source(days=["20061115"]),
    batch_size=8192,
    num_workers=20,
)

# metrics setup
train_metrics = {
    "TEMPERATURE": MetricCollection(
        {
            "mae": re.MeanAbsoluteError(),
        },
    ),
}

val_metrics = {
    "TEMPERATURE": MetricCollection(
        {
            "mae": re.MeanAbsoluteError(),
            "mse": re.MeanSquaredError(),
            "rmse": re.NormalizedRootMeanSquaredError(),
        },
    ),
}

test_metrics = {
    "TEMPERATURE": MetricCollection(
        {
            "mae": re.MeanAbsoluteError(),
            "mse": re.MeanSquaredError(),
            "rmse": re.NormalizedRootMeanSquaredError(),
        },
    ),
}

# define model
model = MLPModel(
    spec=spec,
    train_metrics=train_metrics,
    val_metrics=val_metrics,
    test_metrics=test_metrics,
    feature_paths=nn.ModuleDict(
        {
            # "CH": nn.Sequential(
            #     nn.LazyLinear(1024),
            #     nn.GELU(),
            #     nn.LazyLinear(256),
            #     nn.GELU(),
            #     nn.LazyLinear(128),
            #     nn.GELU(),
            # ),
            "AMPR": nn.Sequential(
                nn.LazyLinear(8),
                nn.GELU(),
            ),
        }
    ),
    output_path=nn.Sequential(
        nn.LazyLinear(128),
        nn.GELU(),
        nn.LazyLinear(128),
        nn.GELU(),
    ),
)

# set up optimizer
opt = torch.optim.NAdam(model.parameters(), lr=0.002)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode="min",
    factor=0.5,
    patience=2,
    threshold=0.001,
    threshold_mode="abs",
)
lr_scheduler_config = {
    "scheduler": lr_scheduler,
    "interval": "epoch",
    "frequency": 1,
    "monitor": "val_loss",
    "strict": True,
}

model.set_optimizer(opt, lr_scheduler_config)

# train!
tracking_uri = "/explore/nobackup/people/dgershm1/mlruns"

with mlf.start_run(
    tracking_uri=tracking_uri,
    experiment_name="+".join(spec.targets.keys()),
) as run:
    mlf_logger = MLFlowLogger(
        tracking_uri=tracking_uri,
        run_id=run.info.run_id,
        log_model="all",
    )

    trainer = L.Trainer(
        max_epochs=25,
        logger=mlf_logger,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5),
            callbacks.LearningRateMonitor(),
            callbacks.RichProgressBar(),
        ],
        enable_progress_bar=True,
        enable_model_summary=True,
        # strategy="deepspeed",
    )

    trainer.fit(model=model, datamodule=datamodule)
